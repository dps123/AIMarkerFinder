import gc
import os
import logging
from random import sample
from string import ascii_letters, digits
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torchmetrics.functional import r2_score
from torch.utils.data import DataLoader, Dataset


def draw_loss_plots(model_name: str, train_loss: list, eval_loss: list, train_class_loss: list, eval_class_loss: list) -> None:
    train_loss[0] = None  # because usually is too big and breaks the scale

    train_loss_chart = plt.plot(train_loss, color="red")
    eval_loss_chart = plt.plot(eval_loss, color="blue")
    train_class_loss_chart = plt.plot(train_class_loss, color="green")
    eval_class_loss_chart = plt.plot(eval_class_loss, color="orange")

    plt.title(f"Training dynamics of {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.legend((train_loss_chart[0], eval_loss_chart[0], train_class_loss_chart[0], eval_class_loss_chart[0]), (
        "training", "evaluation", "train classification", "evaluation classification"))
    plt.grid(True)

    plt.savefig(f"loss_of_{model_name}.pdf")
    plt.clf()


class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        # number of times to allow for no improvement before stopping the execution
        self.patience = patience
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times
    def early_stop_check(self, validation_loss):
        if ((validation_loss + self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((validation_loss + self.min_delta) > self.min_validation_loss):
            # increase the counter if validation loss is not decreased by the min_delta
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CustomDataset(Dataset):
    def __init__(self, X, Xout, Y):
        self.X = X
        self.Xout = Xout
        self.Y = Y  # torch.reshape(Y, (Y.size(dim=0),1))

    def __len__(self):
        return int(self.Y.size(dim=0))

    def __getitem__(self, index):
        return {'X': self.X[index],
                'Xout': self.Xout[index],
                'Y': self.Y[index]
                }


class AEAClassificator(nn.Module):
    def __init__(self, model_name, input_size, bottleneck, class_layer, classes):
        super(AEAClassificator, self).__init__()
        self.model_name = model_name
        self.model_dir = "/tmp/"
        self.name_prefix = ''.join(sample(ascii_letters + digits, 5))
        self.best_model_name = f"{self.model_dir}model_{self.name_prefix}_{self.model_name}.pt"

        self.input_size = input_size

        middle = input_size

        self.attention = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, input_size),
            nn.Softmax(dim=1)
        )

        self.encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, middle),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(middle, bottleneck),
            nn.PReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(bottleneck, class_layer),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(class_layer, classes),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(bottleneck, middle),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(middle, input_size),
            nn.PReLU()
        )

    def __del__(self):
        del self.encoder
        del self.decoder
        del self.classifier
        del self.attention

    def forward(self, x):
        att = self.attention(x)
        x_a = x * (torch.nn.functional.sigmoid(att * 5 - 2.5))

        # Self attention
        # queries = self.query(x)
        # keys = self.key(x)
        # values = self.value(x)
        # scores = queries @ keys.T / (self.input_size ** 0.5)
        # att = self.softmax(scores)
        # x_a = att @ values
        # rr = torch.abs(x - x_a) / torch.norm(x)

        encoded = self.encoder(x_a)
        classified = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return att, encoded, classified, decoded

    def train_model(self, X_train: np.array, X_train_out: np.array, Y_train: np.array, X_test: np.array, X_test_out: np.array, Y_test: np.array,
                    device="cpu",
                    epochs=10000
                    ):
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # nn.BCELoss()
        alpha = 1.0
        beta = 0.3
        gamma = 0.1

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-3,
                                     betas=(0.9, 0.999),
                                     weight_decay=1e-4,
                                     amsgrad=False
                                     )
        # optimizer = Lamb(self.parameters(), lr=3e-2)
        # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters = 5, power = 2.0)

        earlystop = ValidationLossEarlyStopping(patience=100, min_delta=0.005)
        earlystop_class = ValidationLossEarlyStopping(
            patience=100, min_delta=0.005)

        train_data = DataLoader(
            CustomDataset(
                torch.from_numpy(X_train).float().to(device),
                torch.from_numpy(X_train_out).float().to(device),
                torch.from_numpy(Y_train).float().to(device)
            ),
            batch_size=2048, shuffle=True)
        if X_test is not None:
            test_data = DataLoader(
                CustomDataset(
                    torch.from_numpy(X_test).float().to(device),
                    torch.from_numpy(X_test_out).float().to(device),
                    torch.from_numpy(Y_test).float().to(device)
                ), batch_size=2048, shuffle=True)
        else:
            test_data = None

        # train
        train_class_loss, train_loss, eval_loss, eval_class_loss = [], [], [], []
        best_eval_class_loss = 1.0e+5

        for epoch in range(epochs):

            # train the model
            self.train()
            train_loss_epoch, train_class_loss_epoch = [], []

            for batch in train_data:

                _, _, cls, output = self.forward(batch['X'])

                # comparison — forward
                loss_train_value = mse_loss(output, batch['Xout'])
                loss_classifier_value = bce_loss(cls, batch['Y'])
                loss_r2 = 1 - r2_score(output, batch['Xout'])
                loss = alpha * loss_train_value + beta * loss_classifier_value + gamma * loss_r2

                # change weights — backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                train_loss_epoch.append(
                    loss_train_value.detach().cpu().numpy())
                train_class_loss_epoch.append(
                    loss_classifier_value.detach().cpu().numpy())

            train_loss.append(np.mean(train_loss_epoch))
            train_class_loss.append(np.mean(train_class_loss_epoch))

            if test_data is not None:
                # evaluate the model
                self.eval()
                eval_loss_epoch, eval_class_loss_epoch = [], []

                with torch.no_grad():
                    for batch in test_data:

                        _, _, cls, output = self.forward(batch['X'])

                        loss_eval_value = mse_loss(output, batch['Xout'])
                        loss_classifier_value = bce_loss(cls, batch['Y'])

                        eval_loss_epoch.append(
                            loss_eval_value.detach().cpu().numpy())
                        eval_class_loss_epoch.append(
                            loss_classifier_value.detach().cpu().numpy())

                eval_loss.append(np.mean(eval_loss_epoch))
                eval_class_loss.append(np.mean(eval_class_loss_epoch))
            else:
                eval_loss_epoch = train_loss_epoch
                eval_class_loss = train_class_loss

            logging.debug(
                f"Epoch: {epoch + 1}/{epochs}, "
                f"train_loss: {train_loss[-1]:4f}, "
                f"train_class_loss: {train_class_loss[-1]:4f}, "
                f"eval_loss: {eval_loss[-1]:4f}, "
                f"eval_class_loss: {eval_class_loss[-1]:4f}, "
            )
            if alpha * eval_loss[-1] + beta * eval_class_loss[-1] < best_eval_class_loss:
                best_eval_class_loss = alpha * \
                    eval_loss[-1] + beta * eval_class_loss[-1]
                if best_eval_class_loss < 0.1:
                    # self.best_model_name = f"model_{self.model_name}_epoch_{epoch + 1}_mse_{eval_loss[-1]:4f}_bce_{eval_class_loss[-1]:4f}.pt"
                    self.best_model_name = f"{self.model_dir}model_{self.name_prefix}_{self.model_name}_best.pt"
                    logging.debug(f"Saving model {self.best_model_name}")
                    torch.save(self.state_dict(), self.best_model_name)

            if epoch > 10 and np.mean(train_loss_epoch) < 0.05 and earlystop.early_stop_check(np.mean(eval_loss_epoch)) and earlystop_class.early_stop_check(np.mean(eval_class_loss)):
                break

        # draw_loss_plots(self.model_name, train_loss, eval_loss, train_class_loss, eval_class_loss)

        torch.save(self.state_dict(), self.best_model_name)

    def load(self):
        try:
            if not os.path.exists(self.best_model_name):
                raise FileNotFoundError(f"Файл {self.best_model_name} не найден.")
            logging.debug(f"Loading model {self.best_model_name}")
            self.load_state_dict(torch.load(self.best_model_name, weights_only=True))
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")

    def save_to(self, filename):
        try:
            logging.debug(f"Saving model to {filename}")
            torch.save(self.state_dict(), filename)
        except Exception as e:
            logging.error(f"Ошибка при сохранении модели: {e}")

    def run(self, X):
        all_att = []
        all_encoded = []
        all_classified = []
        all_output = []

        data_loader = DataLoader(X, batch_size=2048, shuffle=False)
        for batch in data_loader:
            att, encoded, classified, output = self.forward(batch)

            # Добавляем результаты в списки
            all_att.append(att.detach().cpu().numpy())
            all_encoded.append(encoded.detach().cpu().numpy())
            all_classified.append(classified.detach().cpu().numpy())
            all_output.append(output.detach().cpu().numpy())

        # Объединяем результаты для всех батчей
        all_att = np.concatenate(all_att, axis=0)
        all_encoded = np.concatenate(all_encoded, axis=0)
        all_classified = np.concatenate(all_classified, axis=0)
        all_output = np.concatenate(all_output, axis=0)

        return all_att, all_encoded, all_classified, all_output


class CustomDatasetAEA(Dataset):
    def __init__(self, X, Xout):
        self.X = X
        self.Xout = Xout

    def __len__(self):
        return int(self.Xout.size(dim=0))

    def __getitem__(self, index):
        return {'X': self.X[index],
                'Xout': self.Xout[index]
                }


class AEA(nn.Module):
    def __init__(self, model_name, input_size, bottleneck):
        super(AEA, self).__init__()
        self.model_name = model_name
        self.model_dir = "/tmp/"
        self.name_prefix = ''.join(sample(ascii_letters + digits, 5))
        self.best_model_name = f"{self.model_dir}model_{self.name_prefix}_{self.model_name}.pt"

        self.input_size = input_size

        middle = input_size

        self.attention = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, input_size),
            nn.Softmax(dim=1)
        )

        self.encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, middle),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(middle, bottleneck),
            nn.PReLU()
        )

        self.decoder = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(bottleneck, middle),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(middle, input_size),
            nn.PReLU()
        )

    def __del__(self):
        del self.encoder
        del self.decoder
        del self.attention

    def forward(self, x):
        att = self.attention(x)
        x_a = x * (torch.nn.functional.sigmoid(att * 5 - 2.5))

        encoded = self.encoder(x_a)
        decoded = self.decoder(encoded)
        return att, encoded, decoded

    def train_model(self,
                    X_train: np.array, X_train_out: np.array,
                    X_test: np.array, X_test_out: np.array,
                    device="cpu",
                    epochs=10000
                    ):
        mse_loss = nn.MSELoss()
        alpha = 1.0
        gamma = 0.1

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-3,
                                     betas=(0.9, 0.999),
                                     weight_decay=1e-4,
                                     amsgrad=False
                                     )
        # optimizer = Lamb(self.parameters(), lr=3e-2)
        # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters = 5, power = 2.0)

        earlystop = ValidationLossEarlyStopping(patience=100, min_delta=0.005)

        train_data = DataLoader(
            CustomDatasetAEA(
                torch.from_numpy(X_train).float().to(device),
                torch.from_numpy(X_train_out).float().to(device)
            ),
            batch_size=2048, shuffle=True)
        if X_test is not None:
            test_data = DataLoader(
                CustomDatasetAEA(
                    torch.from_numpy(X_test).float().to(device),
                    torch.from_numpy(X_test_out).float().to(device)
                ), batch_size=2048, shuffle=True)
        else:
            test_data = None

        # train
        train_loss, eval_loss = [], []
        best_eval_loss = 1.0e+5

        for epoch in range(epochs):

            # train the model
            self.train()
            train_loss_epoch = []

            for batch in train_data:

                _, _, output = self.forward(batch['X'])

                # comparison — forward
                loss_train_value = mse_loss(output, batch['Xout'])
                loss_r2 = 1 - r2_score(output, batch['Xout'])
                loss = alpha * loss_train_value + gamma * loss_r2

                # change weights — backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                train_loss_epoch.append(
                    loss_train_value.item())

            train_loss.append(np.mean(train_loss_epoch))

            if test_data is not None:
                # evaluate the model
                self.eval()
                eval_loss_epoch = []

                with torch.no_grad():
                    for batch in test_data:

                        _, _, output = self.forward(batch['X'])

                        loss_eval_value = mse_loss(output, batch['Xout'])

                        eval_loss_epoch.append(
                            loss_eval_value.item())

                eval_loss.append(np.mean(eval_loss_epoch))
            else:
                eval_loss_epoch = train_loss_epoch

            logging.debug(
                f"Epoch: {epoch + 1}/{epochs}, "
                f"train_loss: {train_loss[-1]:4f}, "
                f"eval_loss: {eval_loss[-1]:4f}"
            )
            if alpha * eval_loss[-1] < best_eval_loss:
                best_eval_loss = alpha * eval_loss[-1]
                if best_eval_loss < 0.1:
                    # self.best_model_name = f"model_{self.model_name}_epoch_{epoch + 1}_mse_{eval_loss[-1]:4f}_bce_{eval_class_loss[-1]:4f}.pt"
                    self.best_model_name = f"{self.model_dir}model_{self.name_prefix}_{self.model_name}_best.pt"
                    logging.debug(f"Saving model {self.best_model_name}")
                    torch.save(self.state_dict(), self.best_model_name)

            if epoch > 10 and np.mean(train_loss_epoch) < 0.05 and earlystop.early_stop_check(np.mean(eval_loss_epoch)):
                break
            if epoch > 10 and np.mean(train_loss_epoch) > 10:
                break

        # draw_loss_plots(self.model_name, train_loss, eval_loss, train_class_loss, eval_class_loss)

        torch.save(self.state_dict(), self.best_model_name)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def load(self):
        try:
            if not os.path.exists(self.best_model_name):
                raise FileNotFoundError(f"Файл {self.best_model_name} не найден.")
            logging.debug(f"Loading model {self.best_model_name}")
            self.load_state_dict(torch.load(self.best_model_name, weights_only=True))
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")

    def save_to(self, filename):
        try:
            logging.debug(f"Saving model to {filename}")
            torch.save(self.state_dict(), filename)
        except Exception as e:
            logging.error(f"Ошибка при сохранении модели: {e}")

    def run(self, X):
        all_att = []
        all_encoded = []
        all_output = []

        data_loader = DataLoader(X, batch_size=2048, shuffle=False)
        for batch in data_loader:
            att, encoded, output = self.forward(batch)

            # Добавляем результаты в списки
            all_att.append(att.detach().cpu().numpy())
            all_encoded.append(encoded.detach().cpu().numpy())
            all_output.append(output.detach().cpu().numpy())

        # Объединяем результаты для всех батчей
        all_att = np.concatenate(all_att, axis=0)
        all_encoded = np.concatenate(all_encoded, axis=0)
        all_output = np.concatenate(all_output, axis=0)

        return all_att, all_encoded, all_output
