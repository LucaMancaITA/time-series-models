import tensorflow as tf


def train_one_epoch(model, train_dataset, epoch, loss_function, optimizer):
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_function(y_batch, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_loss += loss.numpy()

        if batch_index % 100 == 99:  # Print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print(f'Batch {batch_index + 1}, Loss: {avg_loss_across_batches:.5f}')

    print()


def validate_one_epoch(model, val_dataset, loss_function):
    running_loss = 0.0

    for x_batch, y_batch in val_dataset:
        predictions = model(x_batch, training=False)
        loss = loss_function(y_batch, predictions)
        running_loss += loss.numpy()

    avg_loss_across_batches = running_loss / len(val_dataset)

    print(f'Val Loss: {avg_loss_across_batches:.5f}')
    print('***************************************************')
    print()
