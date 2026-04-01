{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8b7eb426",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: numpy in c:\\users\\srinidhi\\anaconda3\\lib\\site-packages (2.2.5)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\srinidhi\\anaconda3\\lib\\site-packages (1.6.1)\n",
      "Requirement already satisfied: scipy>=1.6.0 in c:\\users\\srinidhi\\anaconda3\\lib\\site-packages (from scikit-learn) (1.15.3)\n",
      "Requirement already satisfied: joblib>=1.2.0 in c:\\users\\srinidhi\\anaconda3\\lib\\site-packages (from scikit-learn) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in c:\\users\\srinidhi\\anaconda3\\lib\\site-packages (from scikit-learn) (3.5.0)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 25.3 -> 26.0.1\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "!pip install numpy scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e31662d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ae793978",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_data(n_samples):\n",
    "    X = np.random.rand(n_samples, 3)  # features: study hours, sleep, attendance\n",
    "    y = 5*X[:,0] + 3*X[:,1] + 2*X[:,2] + np.random.randn(n_samples)\n",
    "    return X, y\n",
    "\n",
    "# Simulate 3 clients\n",
    "clients_data = [generate_data(100) for _ in range(3)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "71caf803",
   "metadata": {},
   "outputs": [],
   "source": [
    "global_model = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "89af762f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_params(model):\n",
    "    return np.append(model.coef_, model.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1c34ec8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def set_params(model, params):\n",
    "    model.coef_ = params[:-1]\n",
    "    model.intercept_ = params[-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1696d1c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_local_model(X, y, global_params):\n",
    "    local_model = LinearRegression()\n",
    "    \n",
    "    # Initialize with global weights\n",
    "    local_model.coef_ = global_params[:-1]\n",
    "    local_model.intercept_ = global_params[-1]\n",
    "    \n",
    "    # Train locally\n",
    "    local_model.fit(X, y)\n",
    "    \n",
    "    return get_params(local_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "93a32712",
   "metadata": {},
   "outputs": [],
   "source": [
    "def federated_averaging(client_params):\n",
    "    return np.mean(client_params, axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "fe51115b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "--- Round 1 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 2 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 3 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 4 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 5 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 6 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 7 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 8 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 9 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n",
      "\n",
      "--- Round 10 ---\n",
      "Client 1 trained\n",
      "Client 2 trained\n",
      "Client 3 trained\n",
      "Global model updated\n"
     ]
    }
   ],
   "source": [
    "# Initialize global model with dummy fit\n",
    "X_init, y_init = generate_data(10)\n",
    "global_model.fit(X_init, y_init)\n",
    "\n",
    "global_params = get_params(global_model)\n",
    "\n",
    "rounds = 10\n",
    "\n",
    "for r in range(rounds):\n",
    "    client_params = []\n",
    "    \n",
    "    print(f\"\\n--- Round {r+1} ---\")\n",
    "    \n",
    "    # Each client trains locally\n",
    "    for i, (X, y) in enumerate(clients_data):\n",
    "        updated_params = train_local_model(X, y, global_params)\n",
    "        client_params.append(updated_params)\n",
    "        print(f\"Client {i+1} trained\")\n",
    "    \n",
    "    # Server aggregates\n",
    "    global_params = federated_averaging(client_params)\n",
    "    \n",
    "    # Update global model\n",
    "    set_params(global_model, global_params)\n",
    "    \n",
    "    print(\"Global model updated\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "497d159e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Final MSE: 0.9125904466871628\n"
     ]
    }
   ],
   "source": [
    "X_test, y_test = generate_data(100)\n",
    "\n",
    "preds = global_model.predict(X_test)\n",
    "mse = mean_squared_error(y_test, preds)\n",
    "\n",
    "print(\"\\nFinal MSE:\", mse)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
