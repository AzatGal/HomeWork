import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

random_seed = 200


class TranslationDatasetProcessing():

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = load_dataset("tatoeba", lang1=cfg.languages[0], lang2=cfg.languages[1])

        self.make_tokenizer(cfg.languages)
        train_dataset, test_dataset = train_test_split(self.dataset['train']['translation'], test_size=0.2,
                                                       random_state=random_seed)
        self.tokenize_datasets(train_dataset, test_dataset, cfg.languages)

    def make_tokenizer(self, langs):
        self.tokenizers = {}
        for lang in langs:
            # Combine texts from both languages for tokenizer training
            texts = [example['translation'][lang] for example in self.dataset['train']]

            # Initialize a tokenizer with Byte-Pair Encoding
            self.tokenizers[lang] = Tokenizer(BPE(unk_token="<unk>"))
            self.tokenizers[lang].pre_tokenizer = Whitespace()

            # Initialize the trainer for BPE
            trainer = BpeTrainer(special_tokens=["<unk>", "<pad>", "<start>", "<eos>"])

            # Train the tokenizer
            self.tokenizers[lang].train_from_iterator(texts, trainer)

            self.tokenizers[lang].post_processor = TemplateProcessing(
                single="<start> $A <eos>",
                special_tokens=[
                    ("<start>", self.tokenizers[lang].token_to_id("<start>")),
                    ("<eos>", self.tokenizers[lang].token_to_id("<eos>")),
                ],
            )
            print(f'{lang} {self.tokenizers[lang].get_vocab_size()}')

    def tokenize_datasets(self, train_dataset, test_dataset, langs):
        self.train_dataset, self.test_dataset = {}, {}
        for lang in langs:
            self.train_dataset[lang] = [[self.tokenizers[lang].encode(sentence[lang]).ids] for
                                        sentence in train_dataset]
            self.test_dataset[lang] = [[self.tokenizers[lang].encode(sentence[lang]).ids] for
                                       sentence in test_dataset]

    def decode(self, lang, ids):
        self.tokenizers[lang].decode(ids)

    def encode(self, lang, texts):
        self.tokenizers[lang].encode(texts)


class TranslationDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, translation_dataset_processing, mode='train'):
        self.cfg = cfg
        self.translation_dataset_processing = translation_dataset_processing
        self.dataset = getattr(self.translation_dataset_processing, f'{mode}_dataset')

    def __getitem__(self, item):
        return self.dataset[self.cfg.languages[0]][item][0], self.dataset[self.cfg.languages[1]][item][0]

    def __len__(self):
        return len(self.dataset[self.cfg.languages[0]])

    def collate_fn(self, batch):
        max_length_lang_1 = max(len(sequence[0]) for sequence in batch)
        max_length_lang_2 = max(len(sequence[1]) for sequence in batch)

        padded_batch_lang_1 = [torch.tensor(sequence[0] + [
            self.translation_dataset_processing.tokenizers[self.cfg.languages[0]].token_to_id("<pad>")] * (
                                                    max_length_lang_1 - len(sequence[0])), dtype=int) for sequence in
                               batch]
        padded_batch_lang_2 = [torch.tensor(sequence[1] + [
            self.translation_dataset_processing.tokenizers[self.cfg.languages[1]].token_to_id("<pad>")] * (
                                                    max_length_lang_2 - len(sequence[1])), dtype=int) for sequence in
                               batch]

        transposed_data = list(zip(*batch))
        transposed_data[0] = padded_batch_lang_1
        transposed_data[1] = padded_batch_lang_2

        inp_enc = torch.stack(transposed_data[0], 0)
        inp_dec = torch.stack(transposed_data[1], 0)

        return [inp_enc, inp_dec]


if __name__ == '__main__':
    from config.translation_dataset_cfg import cfg as dataset_cfg
    from torch.utils.data import DataLoader

    translationdatasetprocessing = TranslationDatasetProcessing(dataset_cfg)
    train_dataset = TranslationDataset(dataset_cfg, translationdatasetprocessing)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=train_dataset.collate_fn)
    for i, batch in enumerate(train_dataloader):
        a = 1
