import random

import numpy as np
import torch


class DigitSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.beg_seq = 10
        self.end_seq = 11
        self.pad = 12

    def __getitem__(self, item):
        """
        Генерация одного элемента датасета.

        Args:
            index: Индекс запрашиваемого элемента.

        Returns:
            tuple: Тройка (input_sequence, target_sequence, sequence_length),
                   где input_sequence и target_sequence - это последовательности цифр,
                   а sequence_length - длина последовательности.
        """

        sent_len = random.randint(self.cfg.min_sentence_len, self.cfg.max_sentence_len)
        sentence = np.random.randint(0, 9, sent_len)
        sequence_with_tokens = [self.beg_seq] + list(sentence) + [self.end_seq]
        return sequence_with_tokens, sequence_with_tokens, sent_len + 2

    def __len__(self):
        return self.cfg.dataset_size

    def collate_fn(self, batch):
        """
        Функция для подготовки batch перед подачей в модель.

        В этой функции выполняются следующие шаги:
        1. Вычисление максимальной длины предложения в batch.
        2. Дополнение каждого предложения в batch до максимальной длины для encoder и decoder.
        3. Создание тензоров для входов encoder и decoder, а также длин предложений.

        Args:
            batch (list): Список кортежей, содержащих элементы данных и их длины.

        Returns:
            list: Список тензоров для входов encoder и decoder, а также длин предложений.
        """

        max_len = max(len(b[0]) for b in batch)

        # Дополнение предложений для входов encoder и decoder
        padded_sentences_encoder = [self.pad_sentence(b[0][1:-1], max_len) for b in batch]
        padded_sentences_decoder = [self.pad_sentence(b[0], max_len) for b in batch]

        transposed_data = list(zip(*batch))
        transposed_data[0] = padded_sentences_encoder
        transposed_data[1] = padded_sentences_decoder

        # Создание стеков тензоров для входов encoder и decoder
        inp_enc = torch.stack(transposed_data[0], 0)
        inp_dec = torch.stack(transposed_data[1], 0)

        return [inp_enc, inp_dec, torch.tensor(transposed_data[2])]

    def pad_sentence(self, sentence, max_len):
        """
        Вспомогательная функция для дополнения предложения до максимальной длины.
        Добавляет токены padding в конец предложения, чтобы соответствовать максимальной длине.

        Args:
            sentence (list): Предложение для дополнения.
            max_len (int): Длина, до которой нужно дополнить предложение.

        Returns:
            torch.Tensor: Дополненный тензор предложения.
        """
        padded = torch.full((max_len,), self.pad, dtype=int)
        padded[:len(sentence)] = torch.tensor(sentence, dtype=int)
        return padded

if __name__ == '__main__':
    from config.digits_dataset_cfg import cfg as dataset_cfg
    from config.transformer_cfg import cfg as transformer_cfg
    from torch.utils.data import DataLoader

    train_dataset = DigitSequenceDataset(dataset_cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    for i, batch in enumerate(train_dataloader):
        a=1