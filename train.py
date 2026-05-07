import argparse
import os
from models import model_factory
from utils.data_loader import get_train_val_generators

def train():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de Cataratas")
    parser.add_argument("--model", type=str, default="mobilenet_v2", help="Nombre del modelo")
    parser.add_argument("--data", type=str, required=True, help="Ruta a la carpeta de datos")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # 1. Preparar Datos
    train_gen, val_gen = get_train_val_generators(args.data, batch_size=args.batch_size)

    # 2. Construir Modelo
    model = model_factory(args.model)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. Entrenar
    print(f"Iniciando entrenamiento de {args.model}...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs
    )

    # 4. Guardar
    if not os.path.exists('outputs'): os.makedirs('outputs')
    model.save(f'outputs/{args.model}_final.h5')
    print(f"Modelo guardado en outputs/{args.model}_final.h5")

if __name__ == "__main__":
    train()