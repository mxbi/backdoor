from backdoor.dataset.kmnist import KuzushijiMNIST

ds = KuzushijiMNIST()
data = ds.get_data()

print(data['train'][0][0])

KuzushijiMNIST.save_image(data['train'][0][1], 'test_image1.png')