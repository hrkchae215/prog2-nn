def test_accuracy(model, dataloadr):
    n_corrects = 0

    model.eval()
    for image_batch, labei_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)

        predict_batch = logits_batch.argmax(dim=1)
        n_corrects += (label_batch == predict_batch).sum().item()

    accuracy = n_corrects / len(dataloader.dataset)
    return accuracy        

def train(model, dataloader, loss_fn, optimizer):
    """1 epoch の学習を行う"""
    model.train()
    for image_batch, label_batch in dataloader:
        logits_batch = model(image_batch, label_batch)

        loss = loss_fn(logits_batch, label_batch)


        optimizer.zeros_grad()
        loss.backward()
        optimizer.step()

    return loss.item()      

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')


models.train(model,dataloader_test, loss_fn, optimizer)


acc_test = models.test_accuracy(model,dataloader_test)
print(f'testaccuracy: {acc_test*100:.2f}%')

from torch import nn

class MyModel(nn.Module):
    def __unut__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sepuential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits