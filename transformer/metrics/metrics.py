import Levenshtein


def levenshtein_distance(gt_sentence, predicted_sentence):
    # gt_sentence - исходное (реальное) предложение
    # predicted_sentence - предложение, предсказанное трансформером
    # Удалите все знаки препинания из обеих последовательностей
    # Разделите предложения gt_sentence и predicted_sentence на слова, используя метод split()
    gt_sentence = gt_sentence.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
    predicted_sentence = predicted_sentence.replace(",", "").replace(".", "").replace("!", "").replace("?", "")

    gt_sentence_list = gt_sentence.split()
    predicted_sentence_list = predicted_sentence.split()

    # Если списки gt_sentence_list и predicted_sentence_list имеют разную длину, расстояние Левенштейна рассчитывается следующим образом:
    #    1. Определите минимальную min_list_length и максимальную max_list_length длины этих списков
    #    2. Рассчитайте расстояние Левенштейна по следующей формуле:
    # sum_value = sum([1.0 - (Levenshtein.distance(predicted_word, word) / max(len(word), len(predicted_word)))
    #                  for word, predicted_word in zip(gt_sentence_list[:min_list_length], predicted_sentence_list[:min_list_length])
    #                  ])/max_list_length

    gt_length = len(gt_sentence_list)
    predicted_length = len(predicted_sentence_list)

    min_list_length = min(gt_length, predicted_length)
    max_list_length = max(gt_length, predicted_length)

    sum_value = sum([1.0 - (Levenshtein.distance(predicted_word, word) / max(len(word), len(predicted_word)))
                     for word, predicted_word in
                     zip(gt_sentence_list[:min_list_length], predicted_sentence_list[:min_list_length])
                     ]) / max_list_length

    return sum_value
