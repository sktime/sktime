from sktime.utils.dependencies import _safe_import  # [sktime] soft-dep isolation

from .preprocessing_utilities import (
    Detrending,
    Embedding,
    PowerTransformer,
    estimate_initial_condition,
)

torch = _safe_import("torch")


class DataPreprocessor:
    """
    Main class for data preprocessing that orchestrates all transformations.
    """

    def __init__(
        self,
        standardize=True,
        power_transform=False,
        detrending=False,
        preprocessing_method="pos_embedding",
    ):
        """
        Initialize the data preprocessor.

        Args:
            standardize: Whether to standardize the data
            power_transform: Whether to apply power transformation
            detrending: Whether to apply exponential detrending
            preprocessing_method: Method for embedding ('pos_embedding', 'zero_embedding',
                                  'delay_embedding', 'delay_embedding_random')
        """
        self.standardize = standardize
        self.power_transform = power_transform
        self.detrending = detrending
        self.preprocessing_method = preprocessing_method

        # Parameters for inverse transformations
        self.detrending_params_list = None
        self.power_transformer = PowerTransformer()
        self.transformation_mean = None
        self.transformation_std = None

        self.context_mean = None
        self.context_std = None

        self.original_context = None
        self.batch_size = None
        self.feature_dim = None

    def _apply_transformations(self, context):
        """
        Apply power transformation and/or detrending to each batch in the context data.

        Args:
            context: Context data tensor of shape (seq_length, batch_size, N_data)

        Returns
        -------
            Transformed context data
        """
        # Store original context for inverse transformations
        self.original_context = context.clone()

        # Before transformations standardize data
        if self.power_transform or self.detrending:
            self.transformation_mean = torch.mean(context, dim=0)
            self.transformation_std = torch.std(context, dim=0)
            context = (
                context - self.transformation_mean.unsqueeze(0)
            ) / self.transformation_std.unsqueeze(0)

        # Apply power transformation for each batch
        if self.power_transform:
            transformed_context = torch.zeros_like(context)

            for b in range(self.batch_size):
                batch_context = context[:, b, :]
                transformed = self.power_transformer.transform(batch_context)
                transformed_context[:, b, :] = transformed

            context = transformed_context

        # Apply detrending for each batch
        if self.detrending:
            detrended_context = torch.zeros_like(context)
            self.detrending_params_list = []

            for b in range(self.batch_size):
                batch_context = context[:, b, :]
                detrended, params = Detrending.apply_detrending(batch_context)
                detrended_context[:, b, :] = detrended
                self.detrending_params_list.append(params)

            context = detrended_context

        return context

    def _apply_transformations_inverse(self, output):
        """
        Apply inverse power transformation and detrending transformations.

        Args:
            output: Model output of shape (T, batch_size, N)

        Returns
        -------
            Output with transformations reversed
        """
        # Apply inverse detrending for each batch
        if self.detrending and self.detrending_params_list is not None:
            for b in range(self.batch_size):
                batch_output = output[:, b, :]
                batch_context = self.original_context[:, b, :]
                batch_output = Detrending.apply_detrending_inverse(
                    batch_context, batch_output, self.detrending_params_list[b]
                )
                output[:, b, :] = batch_output

        # Apply inverse power transformation for each batch
        if self.power_transform:
            for b in range(self.batch_size):
                batch_output = output[:, b, :]
                batch_output = self.power_transformer.inverse_transform(batch_output)
                output[:, b, :] = batch_output

        # Apply inverse standardization if transformation was applied
        if self.transformation_mean is not None and self.transformation_std is not None:
            output = output * self.transformation_std.unsqueeze(
                0
            ) + self.transformation_mean.unsqueeze(0)

        return output

    def _standardize_data(self, context):
        """
        Standardize each batch in the context data.

        Args:
            context: Context data tensor of shape (seq_length, batch_size, N_data)
            initial_x: Optional initial condition of shape (batch_size, N_data)

        Returns
        -------
            Standardized context and initial_x (if provided)
        """
        if not self.standardize:
            return context

        # Calculate mean and std across time dimension for each batch separately
        self.context_mean = torch.mean(context, dim=0)  # (batch_size, N_data)
        self.context_std = torch.std(context, dim=0)  # (batch_size, N_data)
        self.context_std = torch.clamp(
            self.context_std, min=1e-6
        )  # Avoid division by zero

        # Standardize using broadcasting
        context = (
            context - self.context_mean.unsqueeze(0)
        ) / self.context_std.unsqueeze(0)

        return context

    def _unstandardize_data(self, output):
        """
        Undo standardization by applying the inverse transformation.

        Args:
            output: Model output of shape (T, batch_size, N)

        Returns
        -------
            Output with standardization reversed
        """
        if (
            self.standardize
            and self.context_mean is not None
            and self.context_std is not None
        ):
            return output * self.context_std.unsqueeze(0) + self.context_mean.unsqueeze(
                0
            )
        return output

    def _apply_embedding(self, context, model_dim):
        """
        Apply data preprocessing to each batch to reach model dimension.

        Args:
            context: Context data tensor of shape (seq_length, batch_size, N_data)
            model_dim: Target model dimension

        Returns
        -------
            Preprocessed context data tensor
        """
        context_embedded_batch = []

        for b in range(self.batch_size):
            batch_context = context[:, b, :]
            batch_embedded = Embedding.apply_embedding(
                batch_context, model_dim, self.preprocessing_method
            )
            context_embedded_batch.append(batch_embedded)

        # Align sequence lengths across batches
        seq_lengths = [emb.shape[0] for emb in context_embedded_batch]
        min_seq_len = min(seq_lengths)
        context_embedded_batch = [emb[-min_seq_len:] for emb in context_embedded_batch]

        # Stack along batch dimension
        return torch.stack(context_embedded_batch, dim=1)

    def _prepare_initial_condition(self, context_embedded, initial_x, model_dim):
        """
        Prepare initial condition for forecasting.

        Args:
            context_embedded: Preprocessed context data
            initial_x: Optional initial condition
            model_dim: Model dimension

        Returns
        -------
            Initial condition for forecasting

        Raises
        ------
            ValueError: If initial condition is provided with power transformation or detrending enabled
        """
        if initial_x is None:
            # Use last context value for each batch
            return context_embedded[-1]

        # Raise error if initial condition is provided with power transformation or detrending enabled
        if self.power_transform or self.detrending:
            raise ValueError(
                "Using initial conditions with power transformation or detrending is not supported. "
                "Either disable power transformation and detrending or do not provide an initial condition."
            )

        # Process initial conditions for each batch
        initial_x_processed = torch.zeros(
            self.batch_size, model_dim, device=context_embedded.device
        )
        for b in range(self.batch_size):
            batch_initial = initial_x[b]

            # Apply standardization if enabled
            if (
                self.standardize
                and self.context_mean is not None
                and self.context_std is not None
            ):
                batch_initial = (batch_initial - self.context_mean[b]) / (
                    self.context_std[b] + 1e-8
                )

            # If dimensions are smaller than model_dim, estimate full initial condition
            if initial_x.shape[1] < model_dim:
                # Find matching state in context_embedded
                batch_initial = estimate_initial_condition(
                    batch_initial,
                    context_embedded[:, b, :],
                )

            initial_x_processed[b] = batch_initial

        return initial_x_processed

    def preprocess(self, context, model_dim, initial_x=None):
        """
        Apply the complete preprocessing pipeline to the input data.

        Args:
            context: Context data tensor of shape (seq_length, batch_size, N_data) or (seq_length, N_data)
            model_dim: Target model dimension
            initial_x: Optional initial condition of shape (batch_size, N_data) or (N_data,)

        Returns
        -------
            Preprocessed context data and initial condition
        """
        # Store dimensions
        self.batch_size = context.shape[1]
        self.feature_dim = context.shape[2]

        # Apply transformations (power transformation, detrending)
        context = self._apply_transformations(context)

        # Standardize data
        context = self._standardize_data(context)

        # Apply embedding to reach model dimension
        context_embedded = self._apply_embedding(context, model_dim)

        # Prepare initial batch
        initial_condition = self._prepare_initial_condition(
            context_embedded, initial_x, model_dim
        )

        return context_embedded, initial_condition

    def postprocess(self, output):
        """
        Apply inverse transformations to restore original data scaling.

        Args:
            output: Model output of shape (T, batch_size, N)

        Returns
        -------
            Output with inverse transformations applied
        """
        # Undo standardization
        output = self._unstandardize_data(output)

        # Apply inverse transformations (power transformation, detrending)
        output = self._apply_transformations_inverse(output)

        return output
