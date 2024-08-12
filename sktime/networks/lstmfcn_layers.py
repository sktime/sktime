"""Attention Layers used in by the LSTM-FCN Network.

Ported over from sktime-dl.
"""


def make_attention_lstm():
    """Return AttentionLSTM class used by the LSTM-FCN Network."""
    from tensorflow.keras import activations, constraints, initializers, regularizers
    from tensorflow.keras import backend as K

    # from keras.legacy import interfaces
    from tensorflow.keras.layers import RNN, InputSpec, Layer

    def _time_distributed_dense(
        x,
        w,
        b=None,
        dropout=None,
        input_dim=None,
        output_dim=None,
        timesteps=None,
        training=None,
    ):
        """Apply `y . w + b` for every temporal slice y of x.

        # Arguments
            x: input tensor.
            w: weight matrix.
            b: optional bias vector.
            dropout: whether to apply dropout (same dropout mask
                for every temporal slice of the input).
            input_dim: integer; optional dimensionality of the input.
            output_dim: integer; optional dimensionality of the output.
            timesteps: integer; optional number of timesteps.
            training: training phase tensor or boolean.
        # Returns
            Output tensor.
        """
        if not input_dim:
            input_dim = K.shape(x)[2]
        if not timesteps:
            timesteps = K.shape(x)[1]
        if not output_dim:
            output_dim = K.int_shape(w)[1]

        if dropout is not None and 0.0 < dropout < 1.0:
            # apply the same dropout pattern at every timestep
            ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
            dropout_matrix = K.dropout(ones, dropout)
            expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
            x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

        # collapse time dimension and batch dimension together
        x = K.reshape(x, (-1, input_dim))
        x = K.dot(x, w)
        if b is not None:
            x = K.bias_add(x, b)
        # reshape to 3D tensor
        if K.backend() == "tensorflow":
            x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
            x.set_shape([None, None, output_dim])
        else:
            x = K.reshape(x, (-1, timesteps, output_dim))
        return x

    class AttentionLSTMCell(Layer):
        """Long-Short Term Memory unit - with Attention.

        # Arguments
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use
                (see [activations](keras/activations.md)).
                If you pass None, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            recurrent_activation: Activation function to use
                for the recurrent step
                (see [activations](keras/activations.md)).
            attention_activation: Activation function to use
                for the attention step. If you pass None, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
                (see [activations](keras/activations.md)).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
                used for the linear transformation of the inputs.
                (see [initializers](../initializers.md)).
            recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix,
                used for the linear transformation of the recurrent state.
                (see [initializers](../initializers.md)).
            bias_initializer: Initializer for the bias vector
                (see [initializers](../initializers.md)).
            attention_initializer: Initializer for the `attention_kernel` weights
                matrix, used for the linear transformation of the inputs.
                (see [initializers](../initializers.md)).
            use_chrono_initialization: Boolean.
                If True, add 1 to the bias of the forget gate at initialization.
                Setting it to true will also force `bias_initializer="zeros"`.
                This is recommended in [Jozefowicz et al.]
                (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            recurrent_regularizer: Regularizer function applied to
                the `recurrent_kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            bias_regularizer: Regularizer function applied to the bias vector
                (see [regularizer](../regularizers.md)).
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
                (see [regularizer](../regularizers.md)).
            attention_regularizer: Regularizer function applied to
                the `attention_kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix
                (see [constraints](../constraints.md)).
            recurrent_constraint: Constraint function applied to
                the `recurrent_kernel` weights matrix
                (see [constraints](../constraints.md)).
            bias_constraint: Constraint function applied to the bias vector
                (see [constraints](../constraints.md)).
            attention_constraint: Constraint function applied to
                the `attention_kernel` weights matrix
                (see [constraints](../constraints.md)).
            dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the recurrent state.
            return_attention: Returns the attention vector instead of
                the internal state.
        # References
            - [Long short-term memory]
            (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
            (original 1997 paper)
            - [Learning to forget: Continual prediction with LSTM]
            (http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            - [Supervised sequence labeling with recurrent neural networks]
            (http://www.cs.toronto.edu/~graves/preprint.pdf)
            - [A Theoretically Grounded Application of Dropout
            in Recurrent Neural Networks]
            (http://arxiv.org/abs/1512.05287)
            - [Bahdanau, Cho & Bengio (2014),
            "Neural Machine Translation by Jointly Learning to Align and Translate"]
            (https://arxiv.org/pdf/1409.0473.pdf)
            - [Xu, Ba, Kiros, Cho, Courville, Salakhutdinov, Zemel & Bengio (2016)
            "Show, Attend and Tell: Neural Image Caption Generation
             with Visual Attention"]
            (http://arxiv.org/pdf/1502.03044.pdf)
        """

        _tags = {"python_dependencies": "tensorflow"}

        def __init__(
            self,
            units,
            activation="tanh",
            recurrent_activation="hard_sigmoid",
            attention_activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            attention_initializer="orthogonal",
            bias_initializer="zeros",
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            attention_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            attention_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_attention=False,
            implementation=1,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.input_spec = [InputSpec(ndim=2)]
            self.units = units
            self.activation = activations.get(activation)
            self.recurrent_activation = activations.get(recurrent_activation)
            self.attention_activation = activations.get(attention_activation)
            self.use_bias = use_bias

            self.kernel_initializer = initializers.get(kernel_initializer)
            self.recurrent_initializer = initializers.get(recurrent_initializer)
            self.attention_initializer = initializers.get(attention_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.unit_forget_bias = unit_forget_bias

            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)
            self.attention_regularizer = regularizers.get(attention_regularizer)

            self.kernel_constraint = constraints.get(kernel_constraint)
            self.recurrent_constraint = constraints.get(recurrent_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            self.attention_constraint = constraints.get(attention_constraint)

            self.dropout = min(1.0, max(0.0, dropout))
            self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
            self.return_attention = return_attention
            self._dropout_mask = None
            self._recurrent_dropout_mask = None
            self.implementation = implementation
            self.state_spec = [
                InputSpec(shape=(None, self.units)),
                InputSpec(shape=(None, self.units)),
            ]
            self.state_size = (self.units, self.units)

        def build(self, input_shape):
            """Build the AttentionLSTMCell object."""
            if hasattr(self, "timesteps") and self.timesteps is not None:
                self.timestep_dim = self.timesteps
            else:
                self.timestep_dim = 1  # input_shape[0]

            self.input_dim = input_shape[-1]

            self.kernel = self.add_weight(
                shape=(self.input_dim, self.units * 4),
                name="kernel",
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units * 4),
                name="recurrent_kernel",
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint,
            )

            # add attention kernel
            self.attention_kernel = self.add_weight(
                shape=(self.input_dim, self.units * 4),
                name="attention_kernel",
                initializer=self.attention_initializer,
                regularizer=self.attention_regularizer,
                constraint=self.attention_constraint,
            )

            # add attention weights
            # weights for attention model
            self.attention_weights = self.add_weight(
                shape=(self.input_dim, self.units),
                name="attention_W",
                initializer=self.attention_initializer,
                regularizer=self.attention_regularizer,
                constraint=self.attention_constraint,
            )

            self.attention_recurrent_weights = self.add_weight(
                shape=(self.units, self.units),
                name="attention_U",
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint,
            )

            if self.use_bias:
                if self.unit_forget_bias:

                    def bias_initializer(shape, *args, **kwargs):
                        return K.concatenate(
                            [
                                self.bias_initializer((self.units,), *args, **kwargs),
                                initializers.Ones()((self.units,), *args, **kwargs),
                                self.bias_initializer(
                                    (self.units * 2,), *args, **kwargs
                                ),
                            ]
                        )

                else:
                    bias_initializer = self.bias_initializer
                self.bias = self.add_weight(
                    shape=(self.units * 4,),
                    name="bias",
                    initializer=bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )

                self.attention_bias = self.add_weight(
                    shape=(self.units,),
                    name="attention_b",
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )

                self.attention_recurrent_bias = self.add_weight(
                    shape=(self.units, 1),
                    name="attention_v",
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )
            else:
                self.bias = None
                self.attention_bias = None
                self.attention_recurrent_bias = None

            self.kernel_i = self.kernel[:, : self.units]
            self.kernel_f = self.kernel[:, self.units : self.units * 2]
            self.kernel_c = self.kernel[:, self.units * 2 : self.units * 3]
            self.kernel_o = self.kernel[:, self.units * 3 :]

            self.recurrent_kernel_i = self.recurrent_kernel[:, : self.units]
            self.recurrent_kernel_f = self.recurrent_kernel[
                :, self.units : self.units * 2
            ]
            self.recurrent_kernel_c = self.recurrent_kernel[
                :, self.units * 2 : self.units * 3
            ]
            self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3 :]

            self.attention_i = self.attention_kernel[:, : self.units]
            self.attention_f = self.attention_kernel[:, self.units : self.units * 2]
            self.attention_c = self.attention_kernel[:, self.units * 2 : self.units * 3]
            self.attention_o = self.attention_kernel[:, self.units * 3 :]

            if self.use_bias:
                self.bias_i = self.bias[: self.units]
                self.bias_f = self.bias[self.units : self.units * 2]
                self.bias_c = self.bias[self.units * 2 : self.units * 3]
                self.bias_o = self.bias[self.units * 3 :]
            else:
                self.bias_i = None
                self.bias_f = None
                self.bias_c = None
                self.bias_o = None

            self.built = True

        def _generate_dropout_mask(self, inputs, training=None):
            if 0 < self.dropout < 1:
                ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

                def dropped_inputs():
                    return K.dropout(ones, self.dropout)

                self._dropout_mask = [
                    K.in_train_phase(dropped_inputs, ones, training=training)
                    for _ in range(4)
                ]
            else:
                self._dropout_mask = None

        def _generate_recurrent_dropout_mask(self, inputs, training=None):
            if 0 < self.recurrent_dropout < 1:
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.units))

                def dropped_inputs():
                    return K.dropout(ones, self.dropout)

                self._recurrent_dropout_mask = [
                    K.in_train_phase(dropped_inputs, ones, training=training)
                    for _ in range(4)
                ]
            else:
                self._recurrent_dropout_mask = None

        def call(self, inputs, states, training=None):
            """Call the AttentionLSTMCell."""
            # dropout matrices for input units
            dp_mask = self._dropout_mask
            # dropout matrices for recurrent units
            rec_dp_mask = self._recurrent_dropout_mask

            h_tm1 = states[0]  # previous memory state
            c_tm1 = states[1]  # previous carry state

            # alignment model
            h_att = K.repeat(h_tm1, self.timestep_dim)
            att = _time_distributed_dense(
                inputs,
                self.attention_weights,
                self.attention_bias,
                input_dim=self.input_dim,
                output_dim=self.units,
                timesteps=self.timestep_dim,
            )
            attention_ = self.attention_activation(
                K.dot(h_att, self.attention_recurrent_weights) + att
            )  # energy
            attention_ = K.squeeze(
                K.dot(attention_, self.attention_recurrent_bias), 2
            )  # energy

            alpha = K.exp(attention_)

            if dp_mask is not None:
                alpha *= dp_mask[0]

            alpha /= K.sum(alpha, axis=1, keepdims=True)
            alpha_r = K.repeat(alpha, self.input_dim)
            alpha_r = K.permute_dimensions(alpha_r, (0, 2, 1))

            # make context vector (soft attention after Bahdanau et al.)
            z_hat = inputs * alpha_r
            # context_sequence = z_hat
            z_hat = K.sum(z_hat, axis=1)

            if self.implementation == 1:
                if 0 < self.dropout < 1.0:
                    inputs_i = inputs * dp_mask[0]
                    inputs_f = inputs * dp_mask[1]
                    inputs_c = inputs * dp_mask[2]
                    inputs_o = inputs * dp_mask[3]
                else:
                    inputs_i = inputs
                    inputs_f = inputs
                    inputs_c = inputs
                    inputs_o = inputs
                x_i = K.dot(inputs_i, self.kernel_i)
                x_f = K.dot(inputs_f, self.kernel_f)
                x_c = K.dot(inputs_c, self.kernel_c)
                x_o = K.dot(inputs_o, self.kernel_o)
                if self.use_bias:
                    x_i = K.bias_add(x_i, self.bias_i)
                    x_f = K.bias_add(x_f, self.bias_f)
                    x_c = K.bias_add(x_c, self.bias_c)
                    x_o = K.bias_add(x_o, self.bias_o)

                if 0 < self.recurrent_dropout < 1.0:
                    h_tm1_i = h_tm1 * rec_dp_mask[0]
                    h_tm1_f = h_tm1 * rec_dp_mask[1]
                    h_tm1_c = h_tm1 * rec_dp_mask[2]
                    h_tm1_o = h_tm1 * rec_dp_mask[3]
                else:
                    h_tm1_i = h_tm1
                    h_tm1_f = h_tm1
                    h_tm1_c = h_tm1
                    h_tm1_o = h_tm1
                i = self.recurrent_activation(
                    x_i
                    + K.dot(h_tm1_i, self.recurrent_kernel_i)
                    + K.dot(z_hat, self.attention_i)
                )
                f = self.recurrent_activation(
                    x_f
                    + K.dot(h_tm1_f, self.recurrent_kernel_f)
                    + K.dot(z_hat, self.attention_f)
                )
                c = f * c_tm1 + i * self.activation(
                    x_c
                    + K.dot(h_tm1_c, self.recurrent_kernel_c)
                    + K.dot(z_hat, self.attention_c)
                )
                o = self.recurrent_activation(
                    x_o
                    + K.dot(h_tm1_o, self.recurrent_kernel_o)
                    + K.dot(z_hat, self.attention_o)
                )
            else:
                if 0.0 < self.dropout < 1.0:
                    inputs *= dp_mask[0]
                z = K.dot(inputs, self.kernel)
                if 0.0 < self.recurrent_dropout < 1.0:
                    h_tm1 *= rec_dp_mask[0]
                z += K.dot(h_tm1, self.recurrent_kernel)
                z += K.dot(z_hat, self.attention_kernel)

                if self.use_bias:
                    z = K.bias_add(z, self.bias)

                z0 = z[:, : self.units]
                z1 = z[:, self.units : 2 * self.units]
                z2 = z[:, 2 * self.units : 3 * self.units]
                z3 = z[:, 3 * self.units :]

                i = self.recurrent_activation(z0)
                f = self.recurrent_activation(z1)
                c = f * c_tm1 + i * self.activation(z2)
                o = self.recurrent_activation(z3)

            h = o * self.activation(c)
            if 0 < self.dropout + self.recurrent_dropout:
                if training is None:
                    h._uses_learning_phase = True
            return h, [h, c]

    class AttentionLSTM(RNN):
        """Long-Short Term Memory unit - with Attention.

        # Arguments
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use
                (see [activations](keras/activations.md)).
                If you pass None, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            recurrent_activation: Activation function to use
                for the recurrent step
                (see [activations](keras/activations.md)).
            attention_activation: Activation function to use
                for the attention step. If you pass None, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
                (see [activations](keras/activations.md)).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
                used for the linear transformation of the inputs.
                (see [initializers](../initializers.md)).
            recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix,
                used for the linear transformation of the recurrent state.
                (see [initializers](../initializers.md)).
            bias_initializer: Initializer for the bias vector
                (see [initializers](../initializers.md)).
            attention_initializer: Initializer for the `attention_kernel` weights
                matrix, used for the linear transformation of the inputs.
                (see [initializers](../initializers.md)).
            use_chrono_initialization: Boolean.
                If True, add 1 to the bias of the forget gate at initialization.
                Setting it to true will also force `bias_initializer="zeros"`.
                This is recommended in [Jozefowicz et al.]
                (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            recurrent_regularizer: Regularizer function applied to
                the `recurrent_kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            bias_regularizer: Regularizer function applied to the bias vector
                (see [regularizer](../regularizers.md)).
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
                (see [regularizer](../regularizers.md)).
            attention_regularizer: Regularizer function applied to
                the `attention_kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix
                (see [constraints](../constraints.md)).
            recurrent_constraint: Constraint function applied to
                the `recurrent_kernel` weights matrix
                (see [constraints](../constraints.md)).
            bias_constraint: Constraint function applied to the bias vector
                (see [constraints](../constraints.md)).
            attention_constraint: Constraint function applied to
                the `attention_kernel` weights matrix
                (see [constraints](../constraints.md)).
            dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the recurrent state.
            implementation: Implementation mode, either 1 or 2.
            return_sequences: Boolean. Whether to return the last output.
                in the output sequence, or the full sequence.
            return_state: Boolean. Whether to return the last state
                in addition to the output.
            return_attention: Returns the attention vector instead of
                the internal state.
            go_backwards: Boolean (default False).
                If True, process the input sequence backwards and return the
                reversed sequence.
            stateful: Boolean (default False). If True, the last state
                for each sample at index i in a batch will be used as initial
                state for the sample of index i in the following batch.
            unroll: Boolean (default False).
                If True, the network will be unrolled,
                else a symbolic loop will be used.
                Unrolling can speed-up a RNN,
                although it tends to be more memory-intensive.
                Unrolling is only suitable for short sequences.
        # References
            - [Long short-term memory]
            (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
            (original 1997 paper)
            - [Learning to forget: Continual prediction with LSTM]
            (http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            - [Supervised sequence labeling with recurrent neural networks]
            (http://www.cs.toronto.edu/~graves/preprint.pdf)
            - [A Theoretically Grounded Application of Dropout
              in Recurrent Neural Networks]
            (http://arxiv.org/abs/1512.05287)
            - [Bahdanau, Cho & Bengio (2014)
            "Neural Machine Translation by Jointly Learning to Align and Translate"]
            (https://arxiv.org/pdf/1409.0473.pdf)
            - [Xu, Ba, Kiros, Cho, Courville, Salakhutdinov, Zemel & Bengio (2016)
            "Show, Attend and Tell: Neural Image Caption Generation
             with Visual Attention"]
            (http://arxiv.org/pdf/1502.03044.pdf)
        """

        _tags = {"python_dependencies": "tensorflow"}

        # '@interfaces.legacy_recurrent_support
        def __init__(
            self,
            units,
            activation="tanh",
            recurrent_activation="hard_sigmoid",
            attention_activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            attention_initializer="orthogonal",
            bias_initializer="zeros",
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            attention_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            attention_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            implementation=1,
            return_sequences=False,
            return_state=False,
            return_attention=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            **kwargs,
        ):
            import warnings

            if implementation == 0:
                warnings.warn(
                    "`implementation=0` has been deprecated, "
                    "and now defaults to `implementation=1`."
                    "Please update your layer call.",
                    stacklevel=2,
                )
                implementation = 1

            if K.backend() == "cntk":
                if not kwargs.get("unroll") and (dropout > 0 or recurrent_dropout > 0):
                    warnings.warn(
                        "RNN dropout is not supported with the CNTK backend "
                        "when using dynamic RNNs (i.e. non-unrolled). "
                        "You can either set `unroll=True`, "
                        "set `dropout` and `recurrent_dropout` to 0, "
                        "or use a different backend.",
                        stacklevel=2,
                    )
                    dropout = 0.0
                    recurrent_dropout = 0.0

            cell = AttentionLSTMCell(
                units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                attention_activation=attention_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                attention_initializer=attention_initializer,
                bias_initializer=bias_initializer,
                unit_forget_bias=unit_forget_bias,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                attention_regularizer=attention_regularizer,
                kernel_constraint=kernel_constraint,
                recurrent_constraint=recurrent_constraint,
                bias_constraint=bias_constraint,
                attention_constraint=attention_constraint,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_attention=return_attention,
                implementation=implementation,
            )
            super().__init__(
                cell,
                return_sequences=return_sequences,
                return_state=return_state,
                go_backwards=go_backwards,
                stateful=stateful,
                unroll=unroll,
                **kwargs,
            )
            self.return_attention = return_attention

        def build(self, input_shape):
            """Build the AttentionLSTM object."""
            self.cell.timesteps = input_shape[1]
            self.cell.build(input_shape)

        def call(self, inputs, mask=None, training=None, initial_state=None):
            """Call the AttentionLSTM object."""
            self.cell._generate_dropout_mask(inputs, training=training)
            self.cell._generate_recurrent_dropout_mask(inputs, training=training)
            return super().call(
                inputs, mask=mask, training=training, initial_state=initial_state
            )

        @property
        def units(self):
            """Return property units."""
            return self.cell.units

        @property
        def activation(self):
            """Return property activation."""
            return self.cell.activation

        @property
        def recurrent_activation(self):
            """Return property recurrent_activation."""
            return self.cell.recurrent_activation

        @property
        def attention_activation(self):
            """Return property attention_activation."""
            return self.cell.attention_activation

        @property
        def use_bias(self):
            """Return property use_bias."""
            return self.cell.use_bias

        @property
        def kernel_initializer(self):
            """Return property kernel_initializer."""
            return self.cell.kernel_initializer

        @property
        def recurrent_initializer(self):
            """Return property recurrent_initializer."""
            return self.cell.recurrent_initializer

        @property
        def attention_initializer(self):
            """Return property attention_initializer."""
            return self.cell.attention_initializer

        @property
        def bias_initializer(self):
            """Return property bias_initializer."""
            return self.cell.bias_initializer

        @property
        def unit_forget_bias(self):
            """Return property unit_forget_bias."""
            return self.cell.unit_forget_bias

        @property
        def kernel_regularizer(self):
            """Return property kernel_regularizer."""
            return self.cell.kernel_regularizer

        @property
        def recurrent_regularizer(self):
            """Return property recurrent_regularizer."""
            return self.cell.recurrent_regularizer

        @property
        def bias_regularizer(self):
            """Return property bias_regularizer."""
            return self.cell.bias_regularizer

        @property
        def activity_regularizer(self):
            """Return property activity_regularizer."""
            return self.cell.activity_regularizer

        @property
        def attention_regularizer(self):
            """Return property attention_regularizer."""
            return self.cell.attention_regularizer

        @property
        def kernel_constraint(self):
            """Return property kernel_constraint."""
            return self.cell.kernel_constraint

        @property
        def recurrent_constraint(self):
            """Return property recurrent_constraint."""
            return self.cell.recurrent_constraint

        @property
        def bias_constraint(self):
            """Return property bias_constraint."""
            return self.cell.bias_constraint

        @property
        def attention_constraint(self):
            """Return property attention_constraint."""
            return self.cell.attention_constraint

        @property
        def dropout(self):
            """Return property dropout."""
            return self.cell.dropout

        @property
        def recurrent_dropout(self):
            """Return property recurrent_dropout."""
            return self.cell.recurrent_dropout

        @property
        def implementation(self):
            """Return property implementation."""
            return self.cell.implementation

        def get_config(self):
            """Return configuration dict of the AttentionLSTM object."""
            config = {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "recurrent_activation": activations.serialize(
                    self.recurrent_activation
                ),
                "attention_activation": activations.serialize(
                    self.attention_activation
                ),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "recurrent_initializer": initializers.serialize(
                    self.recurrent_initializer
                ),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "attention_initializer": initializers.serialize(
                    self.attention_initializer
                ),
                "use_chrono_initialization": self.unit_forget_bias,
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "recurrent_regularizer": regularizers.serialize(
                    self.recurrent_regularizer
                ),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "attention_regularizer": regularizers.serialize(
                    self.attention_regularizer
                ),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "recurrent_constraint": constraints.serialize(
                    self.recurrent_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "attention_constraint": constraints.serialize(
                    self.attention_constraint
                ),
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
                "return_attention": self.return_attention,
            }
            base_config = super().get_config()
            del base_config["cell"]
            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            """Create a new AttentionLSTM object from a configuration dict."""
            if "implementation" in config and config["implementation"] == 0:
                config["implementation"] = 1
            return cls(**config)

    return AttentionLSTM
