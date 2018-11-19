import repo as Repo
import settings
if __name__ == '__main__':
    #convert and store train
    trainJson = Repo.DataLoader(dataPath=settings.DATA_TRAIN_PATH).get_json()
    Repo.FileStore(filePath=settings.DATA_TRAIN_JSON, data=trainJson).store_json()
    #convert and store test
    testJson = Repo.DataLoader(dataPath=settings.DATA_TEST_PATH).get_json()
    Repo.FileStore(filePath=settings.DATA_TEST_JSON, data=trainJson).store_json()

