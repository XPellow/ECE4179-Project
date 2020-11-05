import numpy as np
import matplotlib.pyplot as plt

# Setting up vars
filepath = "Gen70\\"
ext = ".npy"
gens = ["10", "20", "30", "40", "50", "60", "70"]

fitness = []
gpool = []
models = []
test_acc = []
test_loss = []
train_acc = []
train_loss = []

lfitness = False
lgpool = False
lmodels = False
ltest_acc = True
ltest_loss = True
ltrain_acc = True
ltrain_loss = True

# Loading data
for gen in gens:
	if lfitness: fitness.append(np.load(filepath+"fitness"+gen+ext))
	if lgpool: gpool.append(np.load(filepath+"gpool"+gen+ext))
	if lmodels: models.append(np.load(filepath+"models"+gen+ext))
if ltest_acc: test_acc = np.load(filepath+"test_acc"+gens[-1]+ext)
if ltest_loss: test_loss = np.load(filepath+"test_loss"+gens[-1]+ext)
if ltrain_acc: train_acc = np.load(filepath+"train_acc"+gens[-1]+ext)
if ltrain_loss: train_loss = np.load(filepath+"train_loss"+gens[-1]+ext)

# Postprocess data

ave_ep = lambda x: np.average(x, axis=1).T
ave_gen = lambda x: np.average(x[:,:,-1], axis=1)
max_gen = lambda x: np.max(x[:,:,-1], axis=1)

test_acc = np.array([i for i in test_acc if i])
test_loss = np.array([i for i in test_loss if i])
train_acc = np.array([i for i in train_acc if i])
train_loss = np.array([i for i in train_loss if i])

test_acc_ep = ave_ep(test_acc)
test_loss_ep = ave_ep(test_loss)
train_acc_ep = ave_ep(train_acc)
train_loss_ep = ave_ep(train_loss)

test_acc_gen_max = max_gen(test_acc)
test_acc_gen_ave = ave_gen(test_acc)
train_acc_gen_max = max_gen(train_acc)
train_acc_gen_ave = ave_gen(train_acc)

print(test_acc.shape)
print(test_acc_ep.shape)
print(test_acc_gen_max.shape)
print(test_acc_gen_ave.shape)

#raise Exception

# Setting up graphs

gen_idx = [0, 10, 20, 30, 40, 50, 60, 69]

legend = ["Generation {}".format(i) for i in gen_idx]

# Graphing epoch on x

plt.figure(figsize = (10, 10))

plt.subplot(2,2,1)
plt.plot(train_loss_ep[:, gen_idx])
plt.title('Model Loss On Training Dataset Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training data loss')
plt.legend(legend)

plt.subplot(2,2,2)
plt.plot(test_loss_ep[:, gen_idx])
plt.title('Model Loss On Testing Dataset Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Testing data loss')
plt.legend(legend)

plt.subplot(2,2,3)
plt.plot(train_acc_ep[:, gen_idx])
plt.title('Model Accuracy On Training Dataset')
plt.xlabel('Epoch')
plt.ylabel('Training data Accuracy')
plt.legend(legend)

plt.subplot(2,2,4)
plt.plot(test_acc_ep[:, gen_idx])
plt.title('Model Accuracy On Testing Dataset')
plt.xlabel('Epoch')
plt.ylabel('Testing data Accuracy')
plt.legend(legend)

#print("Train loss graph dim=" + str(train_loss[:, gen_idx].T.shape))

plt.show()

# Getting generational info



# Graphing generation on x

plt.figure(figsize = (10, 10))

plt.subplot(2,2,1)
plt.plot(train_acc_gen_max)
plt.title('Maximum Model Accuracy on Training Dataset per Generation')
plt.xlabel('Generation')
plt.ylabel('Max Accuracy')

plt.subplot(2,2,2)
plt.plot(train_acc_gen_ave)
plt.title('Average Model Accuracy on Training Dataset per Generation')
plt.xlabel('Generation')
plt.ylabel('Average Accuracy')

plt.subplot(2,2,3)
plt.plot(test_acc_gen_max)
plt.title('Maximum Model Accuracy on Testing Dataset per Generation')
plt.xlabel('Generation')
plt.ylabel('Max Accuracy')

plt.subplot(2,2,4)
plt.plot(test_acc_gen_ave)
plt.title('Average Model Accuracy on Testing Dataset per Generation')
plt.xlabel('Generation')
plt.ylabel('Average Accuracy')

plt.show()