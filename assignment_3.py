import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax
import flax.linen as nn
from jax import lax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)
import streamlit as st


class MLP(nn.Module):
    @nn.compact
    def __call__(self, X, dropout_rate, deterministic):
        X = nn.Dense(256)(X)
        X = nn.relu(X)
        X = nn.Dense(128)(X)
        X = nn.relu(X)
        X = nn.Dense(64)(X)
        X = nn.relu(X)
        X = nn.Dense(32)(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(X)
        y_pred = nn.Dense(1)(X)
        return y_pred

    def loss_fn(self, params, X, y, dropout_rate, deterministic=False, rng=jax.random.PRNGKey(0)):
        y_pred = self.apply(params, X, dropout_rate, deterministic=deterministic, rngs = {'dropout':rng})
        return jnp.sqrt(jnp.mean((y - y_pred)**2))

def fit(model, params, X, y, dropout_rate, deterministic, batch_size=32, learning_rate=0.01, epochs=10, rng=jax.random.PRNGKey(0)):
    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(params)

    loss_fn = partial(model.loss_fn, dropout_rate = dropout_rate, deterministic=deterministic)
    loss_grad_fn = jax.value_and_grad(loss_fn)
    losses = []
    total_epochs = (len(X) // batch_size) * epochs

    carry = {}
    carry["params"] = params
    carry["state"] = opt_state

    @jax.jit
    def one_epoch(carry, rng):
        params = carry["params"]
        opt_state = carry["state"]
        idx = jax.random.choice(rng, jnp.arange(len(X)), shape=(batch_size,), replace=False)
        loss_val, grads = loss_grad_fn(params, X[idx], y[idx], rng=rng)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        carry["params"] = params
        carry["state"] = opt_state

        return carry, loss_val

    carry, losses = lax.scan(one_epoch, carry, jax.random.split(rng, total_epochs))
    return carry["params"], losses

# create dataset

X = np.linspace(0, 10, 500)
eps = np.random.normal(0, 5, 500)
y = X**2 +1 + eps

X = X.reshape(-1,1)
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.fit_transform(X_test)
y_train = scaler_y.fit_transform(y_train)
X_train = jnp.array(X_train).reshape(X_train.shape[0],1)
X_test = jnp.array(X_test).reshape(X_test.shape[0], 1)
y_train = jnp.array(y_train)




# initialize model
# model = MLP()
# params = model.init(jax.random.PRNGKey(0), X_train, 0.2, True)

# Streamlit app
st.title("Dropout Regularization Demo")

st.sidebar.header("Parameters")
dropout_rate = st.sidebar.slider("Dropout rate", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
epochs = st.sidebar.slider("Number of epochs", min_value=1, max_value=100, step=1, value=50)
# dropout_rate = 0
# epochs = 20

model = MLP()
params = model.init(jax.random.PRNGKey(0), X_train, dropout_rate, True)

#train model
params, losses = fit(model, params, X_train, y_train, deterministic = False, dropout_rate = dropout_rate, batch_size=10, learning_rate=0.0001, epochs=epochs, rng=jax.random.PRNGKey(0))

#plot the training loss
fig, ax = plt.subplots()
ax.plot(losses)
ax.set(title="Training Loss", xlabel="Iterations", ylabel="RMSE Loss")
st.pyplot(fig)

#make predictions on test set
test_preds = model.apply(params, X_test, dropout_rate = dropout_rate, deterministic=True, rngs={'dropout': jax.random.PRNGKey(0)})
y_hat = scaler_y.inverse_transform(test_preds)

#calculate RMSE on test set
test_rmse = jnp.sqrt(jnp.mean((y_hat - y_test) ** 2))
print(test_rmse)
st.write(f"Test RMSE: {test_rmse}")

#plot the test predictions
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, label="Ground Truth")
ax.scatter(X_test, y_hat, label="Predictions")
ax.legend()
ax.set(title="Test Predictions", xlabel="X", ylabel="y")
st.pyplot(fig)

dropout_rates = [0, 0.05,0.1, 0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95]

train_losses = []
test_losses = []
train_predictions = []
test_predictions = []
for dropout_rate in dropout_rates:
    # initialize and train model
    model = MLP()
    params = model.init(jax.random.PRNGKey(0), X_train, dropout_rate, True)
    train_params, train_losses_i = fit(model, params, X_train, y_train, deterministic=False, dropout_rate=dropout_rate, batch_size=10, learning_rate=0.0001, epochs=epochs, rng=jax.random.PRNGKey(0))
    
    # evaluate training and test losses and predictions
    train_preds = model.apply(train_params, X_train, dropout_rate=dropout_rate, deterministic=False, rngs={'dropout': jax.random.PRNGKey(1)})
    test_preds = model.apply(train_params, X_test, dropout_rate=dropout_rate, deterministic=True, rngs={'dropout': jax.random.PRNGKey(1)})
    y_hat_train = scaler_y.inverse_transform(train_preds)
    y_hat_test = scaler_y.inverse_transform(test_preds)
    train_loss_i = jnp.sqrt(jnp.mean((y_hat_train - y_train) ** 2))
    test_loss_i = jnp.sqrt(jnp.mean((y_hat_test - y_test) ** 2))
    
    # store losses and predictions
    train_losses.append(train_loss_i)
    test_losses.append(test_loss_i)
    train_predictions.append(y_hat_train)
    test_predictions.append(y_hat_test)

# calculate mean and standard deviation of training and test losses and predictions for each dropout rate
train_losses_mean = np.mean(train_losses)
train_losses_std = np.std(train_losses)
test_losses_mean = np.mean(test_losses)
test_losses_std = np.std(test_losses)
train_predictions_mean = np.mean(train_predictions, axis=1)
test_predictions_mean = np.mean(test_predictions, axis=1)

# calculate bias and variance for each dropout rate
bias = [np.mean((y_train[i] -train_predictions_mean[i])**2 ) for i in range(len(train_predictions_mean))]
variance = [np.var(train_predictions[i]) for i in range(len(train_predictions_mean))]

# plot bias and variance against dropout rate
fig, ax = plt.subplots()
ax.plot(dropout_rates, bias, label="Bias")
ax.plot(dropout_rates, variance, label="Variance")
ax.set(title="Bias-Variance Decomposition", xlabel="Dropout Rate", ylabel="Error")
ax.legend()
st.pyplot(fig)

#scale_bias_variance
bias_scaled = (bias-np.min(bias))/(np.max(bias)-np.min(bias))
variance_scaled = (variance-np.min(variance))/(np.max(variance)-np.min(variance))

fig, ax = plt.subplots()
ax.plot(dropout_rates, bias_scaled, label="Bias")
ax.plot(dropout_rates, variance_scaled, label="Variance")
ax.set(title="Bias-Variance Decomposition (scaled)", xlabel="Dropout Rate", ylabel="Error")
ax.legend()
st.pyplot(fig)
