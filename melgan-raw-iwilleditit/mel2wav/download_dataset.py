import mirdata

print(mirdata.list_datasets())
gtzan = mirdata.initialize("gtzan_genre", data_home="C:\\Users\\oskbs\\Studia\\mir\\melgan-neurips\\data\\raw")
gtzan.download()