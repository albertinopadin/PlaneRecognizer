import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from src.ImageObjectDetectors.TorchConvNet import TorchConvNet
from src.Common import test_training_utils as ttu
import numpy as np


class TorchCNN4Images:
    MAX_INPUT_SIZE = 353894400     # Input size should not be bigger than ~3.5GB (Empirical observation)

    def __init__(self, input_shape, n_output, learning_rate):
        self.model = self.construct_model(input_shape, n_output, learning_rate)
        self.device = self.get_device()
        self.model.to(self.device)
        self.learning_rate = learning_rate
        self.n_output = n_output

    def construct_model(self,
                        input_shape,
                        n_output,
                        learning_rate):
        model = TorchConvNet(n_output=n_output)
        return model

    def get_size_np(self, np_arr):
        return np_arr.size * np_arr.itemsize

    def convert_np_to_torch(self, x):
        x = np.moveaxis(x, -1, 1)  # Change from channels last to channels first
        return torch.from_numpy(x)

    def convert_numpy_images_to_torch_tensor(self, x):
        np_img_arr_size = self.get_size_np(x)
        # TODO: This works, but try to see if can be simplified by using the Torch DataLoader class...
        if np_img_arr_size > TorchCNN4Images.MAX_INPUT_SIZE:
            # Break up input into MAX_INPUT_SIZE chunks:
            n_chunks = int(np_img_arr_size / TorchCNN4Images.MAX_INPUT_SIZE) + 1
            x_chunks = np.array_split(x, n_chunks)
            print(f"Num of chunks img np array split into: {n_chunks}")
            torch_img_tensor = None
            for chunk in x_chunks:
                torch_chunk = self.convert_np_to_torch(chunk)
                torch_chunk = torch_chunk.to(self.device)
                if torch_img_tensor is None:
                    torch_img_tensor = torch_chunk
                else:
                    torch_img_tensor = torch.cat([torch_img_tensor, torch_chunk])
            return torch_img_tensor
        else:
            return self.convert_np_to_torch(x)

    def get_device(self):
        if torch.cuda.is_available():
            print("Running in GPU (CUDA)")
            return torch.device("cuda:0")
        else:
            print("Running in CPU (No CUDA device available)")
            return torch.device("cpu")

    def train(self,
              train_images,
              train_labels,
              batch_size,
              n_epochs,
              train_validation_split=0.2):
        X_train, X_val, y_train, y_val = ttu.split_train_validation_data(train_images,
                                                                         train_labels,
                                                                         train_validation_split)

        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        loss_function = MSELoss()
        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = self.convert_numpy_images_to_torch_tensor(X_train[i: i+batch_size])
                batch_Y = torch.from_numpy(y_train[i: i+batch_size])
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)

                self.model.zero_grad()
                outputs = self.model(batch_X)
                loss = loss_function(outputs, batch_Y)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            print(f"Epoch: {epoch}, Loss: {total_loss}")
        validation_loss, validation_accuracy = self.validate(X_val, y_val, loss_function)
        print(f"Validation loss: {validation_loss}; Validation accuracy: {validation_accuracy}")

    def get_one_hot_predictions(self, predictions):
        predictions_indices = np.argmax(predictions, axis=1)
        return np.eye(self.n_output)[predictions_indices]

    def validate(self, X_val, y_val, loss_function):
        X_val = self.convert_numpy_images_to_torch_tensor(X_val)
        y_val = torch.from_numpy(y_val)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
        validation_loss = loss_function(outputs, y_val)
        predictions = self.get_one_hot_predictions(outputs.cpu().numpy())
        y_val = y_val.cpu().numpy()
        accuracy = (predictions == y_val).mean()
        return validation_loss, accuracy

    def split_gpu_input(self, gpu_input, splits=10):
        return torch.chunk(gpu_input, splits)

    def predict(self, input_data, flatten_output=False, one_hot=False):
        print('[TORCH] In predict method...')
        input_data = self.convert_numpy_images_to_torch_tensor(input_data)

        # Set dropout and batch norm layers to evaluation mode before running inference:
        self.model.eval()
        with torch.no_grad():
            input_chunks = self.split_gpu_input(input_data)
            predictions = None
            for chunk in input_chunks:
                if predictions is None:
                    predictions = self.model(chunk)
                else:
                    predictions = torch.cat([predictions, self.model(chunk)])
            # predictions = self.model(input_data)
        # TODO: do any transformations needed on raw predictions:
        predictions = predictions.cpu().numpy()
        if one_hot:
            predictions = self.get_one_hot_predictions(predictions)
        if flatten_output:
            predictions = predictions.flatten()
        return predictions

    def save_model(self, model_filename):
        model_filename += '_torch.pt'  # Specify using Torch
        torch.save(self.model.state_dict(), model_filename)

    def load_model(self, model_filename):
        model_filename += '_torch.pt'  # Specify using Torch
        state_dict = torch.load(model_filename)
        self.model.load_state_dict(state_dict)
