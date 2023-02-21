import os


class CreateMetadata:
    def __init__(self, user_id):
        self.user_id = user_id

    def create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def write_metadata(self):
        # train metadata 작성
        f = open("./vc/Data/p" + str(self.user_id) + "train_list.txt", 'a')  # test 아닐 때는'w'-> 'a'로 수정해야함
        for i in range(1, 71):
            data = "\n./vc/Data/p%s/%d.wav|20" % (self.user_id, i)
            f.write(data)
        f.close()

        # validation metadata 작성
        f = open("./vc/Data/p" + str(self.user_id) + "val_list.txt", 'a')
        for i in range(70, 81):
            data = "\n./vc/Data/p%s/%d.wav|20" % (self.user_id, i)
            f.write(data)
        f.close()