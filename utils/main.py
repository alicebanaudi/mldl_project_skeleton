import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model_1 import CustomNet  # Importa il modello
from train import train
from eval import validate
from dataset import get_dataloaders  # Supponiamo che tu abbia una funzione per caricare i dati
import argparse
import wandb

def main():
    parser = argparse.ArgumentParser(description="Training di un modello su Tiny-Imagenet")
    parser.add_argument('--model', type=str, default='CustomNet', help='Nome del modello (es: CustomNet)')
    parser.add_argument('--epochs', type=int, default=10, help='Numero di epoche di training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum per SGD')
    args = parser.parse_args()
    
    # Inizializza WandB
    wandb.init(project="tiny-imagenet-training")
    wandb.config.update(args)
    
    # Caricamento dati
    train_loader, val_loader = get_dataloaders()
    
    # Selezione del modello
    if args.model == 'CustomNet':
        model = CustomNet().cuda()
    else:
        raise ValueError(f"Modello {args.model} non supportato.")
    
    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    best_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"Iniziato training n: {epoch}")
        train_loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer)
        print(f"Finito training n: {epoch}")
        
        # Validazione
        val_accuracy = validate(model, val_loader, criterion)
        
        # Logga le metriche su WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })
        
        # Aggiornamento della migliore accuratezza
        best_acc = max(best_acc, val_accuracy)
    
    print(f'Best validation accuracy: {best_acc:.2f}%')
    wandb.finish()

if __name__ == "__main__":
    main()
