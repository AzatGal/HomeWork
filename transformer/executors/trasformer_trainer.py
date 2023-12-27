import gc
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.digits_dataset_cfg import cfg as dataset_cfg
from config.transformer_cfg import cfg as transformer_cfg
from config.evaluation_cfg import cfg as evaluation_cfg

from dataset.digitseq_dataset import DigitSequenceDataset
from models.transformer import Transformer

from metrics.metrics import levenshtein_distance


class Trainer:
    def __init__(self):
        self.__prepare_data(transformer_cfg)
        self.__prepare_model()
        """
        self.logger = Logger(env_path="/Users/azatgalautdinov/Desktop/ML_Homework2/Perceptron/api_token.env",
                             project='azat.galyautdinov161002/ML-Homework1')
                             """

    def __prepare_data(self, dataset_cfg):
        """ Подготовка обучающих и тестовых данных """
        self.train_dataset = DigitSequenceDataset(dataset_cfg)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=transformer_cfg.batch_size, shuffle=False,
                                           collate_fn=self.train_dataset.collate_fn)
        self.test_dataset = DigitSequenceDataset(dataset_cfg)
        self.test_dataset = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                           collate_fn=self.train_dataset.collate_fn)

    def __prepare_model(self):
        """ Подготовка нейронной сети"""
        self.model = Transformer(transformer_cfg).to(transformer_cfg.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.train_dataset.pad)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0, betas=(transformer_cfg.b1, transformer_cfg.b2),
                                          eps=transformer_cfg.eps_opt)

    def save_model(self, filename):
        """
            Сохранение весов модели с помощью torch.save()
            :param filename: str - название файла
            TODO: реализовать сохранение модели по пути os.path.join(evaluation_cfg.exp_dir, f"{filename}.pt")
        """
        # raise NotImplementedError
        save_path = os.path.join(evaluation_cfg.exp_dir, f"{filename}.pt")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, filename):
        """
            Загрузка весов модели с помощью torch.load()
            :param filename: str - название файла
            TODO: реализовать выгрузку весов модели по пути os.path.join(evaluation_cfg.exp_dir, f"{filename}.pt")
        """
        # raise NotImplementedError
        load_path = os.path.join(evaluation_cfg.exp_dir, f"{filename}.pt")
        self.model.load_state_dict(torch.load(load_path))

    def create_masks(self, encoder_input, decoder_input):
        """
        Создает маски для трансформера, которые необходимы для корректной работы внимания (attention) в модели.

        Включает в себя следующие маски:
        1. Маска для padding входа энкодера: Эта маска используется для исключения влияния padding токенов на результаты внимания в энкодере.
        2. Маска для padding входов энкодера-декодера: Применяется в декодере для обеспечения того, чтобы padding
                                                         токены не участвовали в расчетах внимания.
        3. Маска для предотвращения утечки будущей информации в декодере: Эта маска гарантирует, что каждая позиция в
                                                декодере может взаимодействовать только с предшествующими ей позициями,
                                                что предотвращает использование "будущей" информации при генерации текущего токена.

        :param encoder_input: Тензор, представляющий последовательность на вход энкодера.
        :param decoder_input: Тензор, представляющий последовательность на вход декодера.
        :return:
            - Маска для padding входа декодера.
            - Маска для padding входов энкодера-декодера.
            - Маска для предотвращения утечки будущей информации в декодере.
        """
        # TODO написать функцию для создания масок

        encoder_padding_mask = (encoder_input == self.train_dataset.pad).unsqueeze(1).unsqueeze(2)
        decoder_padding_mask = (decoder_input == self.train_dataset.pad).unsqueeze(1).unsqueeze(2)
        decoder_future_mask = torch.triu(torch.ones(decoder_input.size(1), decoder_input.size(1)),
                                         diagonal=1).bool().unsqueeze(0)
        return encoder_padding_mask, decoder_padding_mask, decoder_future_mask

    def make_step(self, batch, update_model = True):
        """
        Выполняет один шаг обучения для модели трансформера.

        Этапы включают:
        1. Forward Pass:
            - Получение входных данных для энкодера и декодера из батча.
            - Получение масок для обработки padding в последовательностях и для предотвращения утечки будущей информации в декодере.
            - Выполнение forward pass модели с данными энкодера и декодера.

        2. Вычисление функции потерь:
            - Функция потерь рассчитывается на основе предсказаний модели и целевых значений из батча.
            - Предсказания модели и целевые значения преобразуются в соответствующие форматы для функции потерь.

        3. Backward Pass и обновление весов:
            - Выполнение backward pass для расчета градиентов.
            - Обновление весов модели с помощью оптимизатора.

            :param update_model:
            :param batch: tuple data - encoder input, decoder input, sequence length
            :return: значение функции потерь, выход модели
            # TODO: реализуйте инференс модели для данных batch, посчитайте значение целевой функции
        """
        # raise NotImplementedError

        encoder_input, decoder_input, seq_len = batch
        encoder_input = encoder_input.to(transformer_cfg.device)
        decoder_input = decoder_input.to(transformer_cfg.device)

        self.optimizer.zero_grad()

        encoder_padding_mask, decoder_padding_mask, decoder_future_mask = self.create_masks(encoder_input,
                                                                                            decoder_input)

        output = self.model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask,
                            decoder_future_mask)

        loss = self.criterion(output.view(-1, output.size(-1)), decoder_input.view(-1))

        if update_model:
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), output

    def train_epoch(self, *args, **kwargs):
        """
            Обучение модели на self.train_dataloader в течение одной эпохи. Метод проходит через все обучающие данные и
            вызывает метод self.make_step() на каждом шаге.

            TODO: реализуйте функцию обучения с использованием метода self.make_step(batch, update_model=True),
                залогируйте на каждом шаге значение целевой функции, accuracy, расстояние Левенштейна.
                Не считайте токены padding при подсчете точности
        """
        self.model.train()
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        total_distance = 0
        total_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            loss, output = self.make_step(batch, update_model=True)
            total_loss += loss
            total_batches += 1

            # Calculate accuracy and Levenshtein distance
            target = batch[1].to(transformer_cfg.device)
            accuracy = (output.argmax(dim=-1) == target).sum().item() / (target != self.train_dataset.pad).sum().item()
            distance = levenshtein_distance(output.argmax(dim=-1), target)

            total_accuracy += accuracy
            total_distance += distance

            print(
                f"Batch {batch_idx + 1}/{len(self.train_dataloader)} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f} - Levenshtein Distance: {distance}")

        avg_loss = total_loss / total_batches
        avg_accuracy = total_accuracy / total_batches
        avg_distance = total_distance / total_batches

        print(
            f"Epoch finished - Average Loss: {avg_loss:.4f} - Average Accuracy: {avg_accuracy:.4f} - Average Levenshtein Distance: {avg_distance:.4f}")


    def evaluate(self, *args, **kwargs):
        """
        Метод для оценки  модели трансформера.

        Основные шаги процесса инференса включают:
        1. Перевод модели в режим оценки (`model.eval()`), что отключает слои, работающие по-разному во время обучения и инференса (например, Dropout).
        2. Перебор данных по батчам: для каждого батча последовательно генерируются предсказания.
        3. Инференс в цикле:
            a. В качестве входа энкодера на каждом шаге используется весь input экодера, также как и на этапе обучения.
            b. В качестве входа декодера на первом шаге цикла подается одним токен - self.train_dataset.beg_seq
               Пока модель не предскажет токен конца последовательности (self.train_dataset.end_seq) или количество итераций цикла достигнет
               максимального значения transformer_cfg.max_search_len,
               на каждом шаге происходит следующее:
               - Модель получает на вход текущую последовательность и выдает предсказания для следующего токена.
               - Из этих предсказаний выбирается токен с наибольшей вероятностью (используется argmax).
               В домашнем задании с обучением перевода добавьте  softmax с температурой и вероятностное сэмплирование.
               - Этот токен добавляется к текущей последовательности декодера, и процесс повторяется.
        5. Вычисление метрик(accuracy, расстояние Левенштейна) для сгенерированной последовательности, исключая паддинг-токены из подсчета.

    TODO: Реализуйте функцию оценки должна включать логирование значений функции потерь и точности,
          не учитывайте паддинг-токены при подсчете точности.

        """
        self.model.eval()
        # raise NotImplementedError

        for batch_idx, batch in enumerate(self.test_dataset):
            encoder_input, decoder_input, seq_len = batch
            encoder_input = encoder_input.to(transformer_cfg.device)
            decoder_input = decoder_input.to(transformer_cfg.device)

            encoder_padding_mask, decoder_padding_mask, decoder_future_mask = self.create_masks(encoder_input,
                                                                                                decoder_input)

            with torch.no_grad():
                output = self.model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask,
                                    decoder_future_mask)

            outputs = torch.argmax(output, dim=1).to(transformer_cfg.device)
            batch['label'] = batch['label'].to(transformer_cfg.device)

            lev_dis = levenshtein_distance(batch['text'], outputs)

            # TODO: Log the loss function and metrics (accuracy, Levenshtein distance)

            del batch
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

    def fit(self, *args, **kwargs):
        """
            Основной цикл обучения модели. Данная функция должна содержать один цикл на заданное количество эпох.
            На каждой эпохе сначала происходит обучение модели на обучающих данных с помощью метода self.train_epoch(),
            а затем оценка производительности модели на тестовых данных с помощью метода self.evaluate()

            # TODO: реализуйте основной цикл обучения модели, сохраните веса модели с лучшим значением accuracy на
                тестовой выборке
        """
        # raise NotImplementedError
        for epoch in range(transformer_cfg.num_epochs):
            print(f"Epoch {epoch + 1}/{transformer_cfg.num_epochs}")
            self.train_epoch()
            self.evaluate()
        self.evaluate()


if __name__ == '__main__':

    trainer = Trainer()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
