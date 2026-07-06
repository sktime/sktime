from sktime.utils.dependencies import _safe_import  # [sktime] soft-dep isolation

from .preprocessing import DataPreprocessor

torch = _safe_import("torch")
F = _safe_import("torch.nn.functional")


class DynaMixForecaster:
    """
    Forecasting pipeline for DynaMix models with batch processing support.
    """

    def __init__(self, model):
        """
        Initialize the forecaster with a DynaMix model.

        Args:
            model: DynaMix model instance
        """
        self.model = model

    def _init_latent_state(self, initial_condition):
        """
        Initialize the latent state from the initial condition.

        Args:
            initial_condition: Initial state of shape (batch_size, N)

        Returns
        -------
            Initial latent state z
        """
        N = self.model.N

        # Initialize latent state
        z = torch.matmul(initial_condition, self.model.B).t()  # (M, batch_size)
        z[:N, :] = initial_condition.t()

        return z

    def _reshape_for_model(self, context, initial_x=None, device=None):
        """
        Prepare and reshape input data for the model.
        Handles tensor conversion, dimension adjustments, and reshaping when feature_dim > model_dim.

        Args:
            context: Context data tensor of shape (seq_length, batch_size, feature_dim) or (seq_length, feature_dim)
            initial_x: Optional initial condition of shape (batch_size, feature_dim) or (feature_dim,)
            device: Device to place tensors on

        Returns
        -------
            Processed context, initial_x, dimensions, and reshaping metadata
        """
        # Get the dtype from model parameters
        model_dtype = next(self.model.parameters()).dtype

        # Convert to torch tensor if needed
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=model_dtype, device=device)
        elif context.device != device or context.dtype != model_dtype:
            context = context.to(device=device, dtype=model_dtype)

        if initial_x is not None and not isinstance(initial_x, torch.Tensor):
            initial_x = torch.tensor(initial_x, dtype=model_dtype, device=device)
        elif initial_x is not None and (
            initial_x.device != device or initial_x.dtype != model_dtype
        ):
            initial_x = initial_x.to(device=device, dtype=model_dtype)

        # Check data dimensions and reshape if needed
        original_dim = context.dim()
        if original_dim == 2:
            context = context.unsqueeze(
                1
            )  # (seq_length, feature_dim) -> (seq_length, 1, feature_dim)
        elif original_dim != 3:
            raise ValueError(
                f"Expected 2D or 3D tensor for context, got shape {context.shape} with {context.dim()} dimensions"
            )
        if initial_x is not None and initial_x.dim() == 1:
            initial_x = initial_x.unsqueeze(0)  # (feature_dim,) -> (1, feature_dim)
            if initial_x.shape[1] != context.shape[2]:
                raise ValueError(
                    f"Initial condition has {initial_x.shape[1]} features, but context has {context.shape[2]} features"
                )

        # Data shape
        seq_length, batch_size, feature_dim = context.shape

        # Check if reshaping is needed for model dimension
        if feature_dim <= self.model.N:
            return (
                context,
                initial_x,
                (batch_size, feature_dim, False, None, None, original_dim),
            )

        print(
            f"Warning: Input feature dimension {feature_dim} exceeds model dimension {self.model.N}. "
            f"This may lead to performance degradation."
            f"Reshaping data to treat each feature as separate time series."
        )

        # Store original dimensions for reshaping back later
        original_batch_size = batch_size
        original_feature_dim = feature_dim

        # Reshape context to (seq_length, batch_size * feature_dim, 1)
        transposed = context.permute(0, 2, 1)
        new_batch_size = batch_size * feature_dim
        reshaped_context = transposed.reshape(seq_length, new_batch_size, 1)

        # Similarly reshape initial_x if provided
        reshaped_initial_x = initial_x
        if initial_x is not None:
            # Reshape from (batch_size, feature_dim) to (batch_size * feature_dim, 1)
            reshaped_initial_x = initial_x.transpose(0, 1).reshape(new_batch_size, 1)

        return (
            reshaped_context,
            reshaped_initial_x,
            (
                new_batch_size,
                1,
                True,
                original_batch_size,
                original_feature_dim,
                original_dim,
            ),
        )

    def _reshape_to_original(self, output, reshape_metadata):
        """
        Reshape output back to original dimensions.
        Handles both high-dimensional reshaping and 2D input restoration.

        Args:
            output: Model output of shape (T, batch_size, N)
            reshape_metadata: Tuple containing (was_reshaped, original_batch_size, original_feature_dim, original_dim)

        Returns
        -------
            Output with original shape restored
        """
        _, _, was_reshaped, original_batch_size, original_feature_dim, original_dim = (
            reshape_metadata
        )

        # Step 1: Reshape back to original dimensions if needed
        if was_reshaped:
            # Current shape: (T, batch_size=original_batch_size*original_feature_dim, 1)
            T = output.shape[0]

            # First reshape to (T, original_feature_dim, original_batch_size)
            # by treating the batch dimension as (original_feature_dim, original_batch_size)
            reshaped = output.reshape(T, original_feature_dim, original_batch_size, -1)

            # Then permute to (T, original_batch_size, original_feature_dim)
            output = reshaped.permute(0, 2, 1, 3).squeeze(-1)

        # Step 2: If input was 2D, remove batch dimension from output
        if original_dim == 2 and output.shape[1] == 1:
            output = output.squeeze(1)

        return output

    @torch.no_grad()
    def forecast(
        self,
        context,
        horizon,
        preprocessing_method="pos_embedding",
        standardize=True,
        fit_nonstationary=False,
        initial_x=None,
    ):
        """
        Efficient batched forecasting with the DynaMix model.

        This method implements a complete forecasting pipeline including:
        - Data preprocessing (power transformation, detrending, standardization)
        - Embedding techniques for dimensionality matching
        - DynaMix model prediction
        - Data postprocessing (inverse transformations)

        Args:
            context: Context data tensor of shape (seq_length, batch_size, feature_dim) or (seq_length, feature_dim)
            horizon: Forecast horizon (number of steps to predict)
            preprocessing_method: Data preprocessing method ('pos_embedding', 'zero_embedding',
                                  'delay_embedding', or 'delay_embedding_random') (default: 'pos_embedding')
            standardize: Whether to standardize the data (default: True)
            fit_nonstationary: Whether to fit a non-stationary time series (default: False)
            initial_x: Optional initial condition of shape (batch_size, feature_dim) or (feature_dim,)

        Returns
        -------
            Predicted sequence of shape (horizon, batch_size, feature_dim)
        """
        # Get model dimensions
        M = self.model.M
        N = self.model.N
        device = (
            context.device if isinstance(context, torch.Tensor) else self.model.B.device
        )
        model_dtype = next(self.model.parameters()).dtype

        # Apply context reshaping if needed
        context, initial_x, shape_metadata = self._reshape_for_model(
            context, initial_x, device
        )

        # Create data preprocessor
        preprocessor = DataPreprocessor(
            standardize=standardize,
            power_transform=fit_nonstationary,
            detrending=fit_nonstationary,
            preprocessing_method=preprocessing_method,
        )

        # Step 1: Apply preprocessing pipeline
        context_embedded, initial_condition = preprocessor.preprocess(
            context, self.model.N, initial_x
        )

        # Step 2: Initialize latent state
        z = self._init_latent_state(initial_condition)

        # Step 3: Perform forecasting loop
        Z_gen = torch.empty(
            horizon, M, shape_metadata[0], device=device, dtype=model_dtype
        )
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=device.type == "cuda",
        ):
            precomputed_cnn = self.model.precompute_cnn(context_embedded)
            for t in range(horizon):
                z = self.model(z, context_embedded, precomputed_cnn=precomputed_cnn)
                Z_gen[t] = z

        # Step 4: Apply observation generation
        output = Z_gen[:, : shape_metadata[1], :].permute(
            0, 2, 1
        )  # (horizon, batch_size, feature_dim)

        # Step 5: Apply inverse data transformations (e.g. standardization, ...)
        output = preprocessor.postprocess(output)

        # Step 6: Reshape back to original dimensions if needed
        output = self._reshape_to_original(output, shape_metadata)

        return output
