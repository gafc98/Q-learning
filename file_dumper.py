import matplotlib.pyplot as plt

class Dumper:
    def __init__(self, file_name):
        self.file_name = file_name

    def dump(self, data):
        data = str(data)
        with open(self.file_name, "a") as f:
            f.write(data + '\n')

if __name__ == '__main__':
    file_name = 'data.txt'
    data = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line))
    plt.plot(data)
    plt.show()