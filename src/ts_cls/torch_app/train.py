import time
import torch


def train_val_loop(model, train_dataloader, test_dataloader, optimizer,
                   criterion, max_epochs, device):

    # Training loop
    print("\nTraining loop ...")
    for step in range(max_epochs):

        t0 = time.time()
        train_loss = 0.0
        model.train(True)

        # Loop through all the batches
        for _, batch in enumerate(train_dataloader):

            x_batch, y_batch = batch

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch.float())

            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_dataloader)
        t1 = time.time()

        model.train(False)
        val_loss = 0.0

        for _, batch in enumerate(test_dataloader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_dataloader)

        print(f"Step {step} | train loss {avg_train_loss:.4f}" \
              f"| val_loss {avg_val_loss:.4f} | step time {(t1-t0)*1000:.2f}ms")
