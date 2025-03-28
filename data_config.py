
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'train':
            self.root_dir = 'samples\\train'
        elif data_name == 'test':
            self.root_dir = 'sample\\test'

        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='train')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

