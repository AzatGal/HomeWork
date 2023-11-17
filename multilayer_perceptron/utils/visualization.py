import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Tk

def confusion_matrix(conf_matrix, *args, **kwargs):
    """
        Построение и визуализация confusion matrix
            confusion_matrix - матрица NxN, где N - кол-во классов в наборе данных
            confusion_matrix[i, j] - кол-во элементов класса "i", которые классифицируются как класс "j"

        :return plt.gcf() - matplotlib figure
        TODO: реализуйте построение и визуализацию confusion_matrix, подпишите оси на полученной визуализации, добавьте значение confusion_matrix[i, j] в соотвествующие ячейки на изображении
    """

    # raise NotImplementedError

    plt.cla()
    plt.clf()
    plt.imshow(conf_matrix, cmap='Blues')

    rows, cols = conf_matrix.shape
    for i in range(rows):
        for j in range(cols):
            item = str(
                int(
                    conf_matrix[i][j]
                )
            )
            plt.text(j, i, item, ha='center', va='center')

    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.colorbar()  # label='')

    return plt.gcf()
