from transformer import Transformer
from dataset import build_dataset

if __name__ == "__main__":
    train, test = build_dataset()

    model = Transformer()
    model.adapt(train)
    model.compile(
        optimizer="Adam", loss="cross_entropy", metrics=["cross_entropy", "acc"]
    )
    model.build((None, 512))
    model.summary()

    model.fit(
        train,
        batch_size=32,
        epochs=10,
        verbose=1,
        validation_split=0.2,
    )
