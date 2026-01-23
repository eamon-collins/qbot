# Quoridor Neural Network with Policy and Value heads (AlphaZero-style)
import logging
import argparse
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from StateNode import QuoridorDataset, MultiFileTrainingSampleDataset

# Action space:
#   0-80:    pawn move to square (row * 9 + col)
#   81-144:  horizontal wall at (row * 8 + col)
#   145-208: vertical wall at (row * 8 + col)
NUM_PAWN_ACTIONS = 81   # 9x9 board destinations
NUM_WALL_ACTIONS = 128  # 8x8 * 2 orientations
NUM_ACTIONS = NUM_PAWN_ACTIONS + NUM_WALL_ACTIONS  # 209 total

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Leela Chess Zero variant: Scale + Bias).
    """
    def __init__(self, channels, se_channels=32):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, se_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(se_channels, 2 * channels)

    def forward(self, x):
        batch, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(batch, c)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        
        # Split into scale (w) and bias (bias)
        # Lc0: output = (sigmoid(w) * x) + bias
        w, bias = y.split(c, dim=1)
        
        # Use 'batch' (int) for view, not the tensor
        w = torch.sigmoid(w).view(batch, c, 1, 1)
        bias = bias.view(batch, c, 1, 1)
        
        return (x * w) + bias

class ResidualBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation: conv -> bn -> relu -> conv -> bn -> SE -> skip -> relu"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return F.relu(out)


class QuoridorNet(nn.Module):
    """
    AlphaZero-style network with shared trunk, policy head, and value head.

    The board is always presented from the CURRENT PLAYER's perspective:
    - Channel 0 is always "my" pawn, Channel 1 is always opponent's pawn
    - This means when loading P2's turn, we swap P1/P2 positions and fence counts
    - Value output is always from current player's perspective (+1 = I'm winning)

    Input: (batch, 6, 9, 9) tensor
        [0] Current player's pawn position (one-hot)
        [1] Opponent's pawn position (one-hot)
        [2] Horizontal walls (padded 8x8 -> 9x9)
        [3] Vertical walls (padded 8x8 -> 9x9)
        [4] Current player's fences remaining / 10 (constant plane)
        [5] Opponent's fences remaining / 10 (constant plane)

    Output: (policy_logits, value)
        policy_logits: (batch, 209) raw logits for all actions
        value: (batch, 1) evaluation from current player's view in [-1, 1]
    """

    def __init__(self, num_channels=64, num_blocks=6):
        super().__init__()
        self.num_channels = num_channels
        self.num_blocks = num_blocks

        # Input convolution: 6 channels -> num_channels
        self.input_conv = nn.Conv2d(6, num_channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.residual_tower = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])

        # Policy head: conv 1x1 -> bn -> relu -> flatten -> fc
        # (Kept standard AlphaZero style for spatial preservation)
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 9 * 9, NUM_ACTIONS)

        # Value head: conv 3x3 (32ch) -> bn -> relu -> GAP -> fc(128) -> relu -> fc(1) -> tanh
        # Using GAP allows us to drop the massive input layer (9*9*channels)
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, 6, 9, 9)

        # Input convolution
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.residual_tower:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)  # flatten
        p = self.policy_fc(p)

        # Value head (GAP)
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = F.adaptive_avg_pool2d(v, 1).view(v.size(0), -1) # GAP: (B, 32, 9, 9) -> (B, 32)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

def train_step(model, optimizer, data_batch):
    """
    Single training step with combined policy and value loss.
    """
    model.train()
    states, policy_targets, value_targets = data_batch

    if torch.cuda.is_available():
        states = states.cuda()
        policy_targets = policy_targets.cuda()
        value_targets = value_targets.cuda()

    optimizer.zero_grad()
    policy_logits, values = model(states)

    # Value loss: MSE
    value_loss = F.mse_loss(values, value_targets)

    # Policy loss: Cross-entropy with MCTS visit distribution
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.sum(policy_targets * log_probs, dim=1).mean()

    # Combined loss
    loss = value_loss + policy_loss

    loss.backward()
    optimizer.step()

    return loss.item(), value_loss.item(), policy_loss.item()


def train(model, data_files: str | list[str], batch_size: int, num_epochs: int, stream: bool = False):
    """
    Train the model on training data.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    if torch.cuda.is_available():
        model.cuda()

    # Normalize to list
    if isinstance(data_files, str):
        data_files = [data_files]

    # Check if all files are .qsamples
    all_qsamples = all(f.endswith('.qsamples') for f in data_files)

    if all_qsamples:
        dataset = MultiFileTrainingSampleDataset(data_files, batch_size)
    else:
        logging.warning("This method of training samples is deprecated, quitting")
        sys.exit(-1)

    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=1,
        persistent_workers=False,
        pin_memory=True
    )

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_value_loss = 0
        epoch_policy_loss = 0
        num_batches = 0

        for batch in train_loader:
            loss, v_loss, p_loss = train_step(model, optimizer, batch)
            epoch_loss += loss
            epoch_value_loss += v_loss
            epoch_policy_loss += p_loss
            num_batches += 1

            if num_batches % 100 == 1:
                logging.debug(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss:.4f} (v:{v_loss:.4f} p:{p_loss:.4f})")

        avg_loss = epoch_loss / num_batches
        avg_v_loss = epoch_value_loss / num_batches
        avg_p_loss = epoch_policy_loss / num_batches
        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch}, LR: {current_lr:.6f}  Avg Loss: {avg_loss:.4f} (value:{avg_v_loss:.4f} policy:{avg_p_loss:.4f})")

    ##cleanup
    train_loader._iterator = None
    if hasattr(train_loader, '_workers'):
        for w in train_loader._workers:
            w.terminate()
    del train_loader
    if hasattr(dataset, 'samples') and dataset.samples is not None:
        dataset.samples.clear()
    del dataset
    del optimizer
    del scheduler

def main():
    parser = argparse.ArgumentParser(description='Train Quoridor Policy-Value Network')
    parser.add_argument('--load-tree', type=str, dest="load_tree", help='Path to load search tree')
    parser.add_argument('--load-model', type=str, dest="load_model", help='Path to load model weights')
    parser.add_argument('--save-model', type=str, dest="save_model", help='Path to save model weights')
    parser.add_argument('--export', dest="export", help='Export TorchScript model for C++ inference',
                        action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, dest="batch_size", default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--channels', type=int, default=128, help='Number of channels in residual tower')
    parser.add_argument('--blocks', type=int, default=15, help='Number of residual blocks')
    parser.add_argument('--big-model', dest="big_model", help='Use model with 6m parameters instead of 500k',
                        action='store_true', default=False)
    parser.add_argument('--log-level', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    model = QuoridorNet(num_channels=args.channels, num_blocks=args.blocks)
    logging.info(f"Created model with {args.channels} channels, {args.blocks} residual blocks")

    if args.load_model:
        logging.info(f"Loading weights from {args.load_model}")
        model.load_state_dict(torch.load(args.load_model, weights_only=True))

    if args.export:
        if not args.save_model:
            logging.error("--save-model required with --export")
            sys.exit(1)

        model.eval()

        # Save trainable weights (.model file for continued training)
        weights_path = args.save_model.replace('.pt', '.model')
        if weights_path == args.save_model:
            weights_path = args.save_model + '.model'
        torch.save(model.state_dict(), weights_path)
        logging.info(f"Saved trainable weights to {weights_path}")

        # Export TorchScript model (.pt file for C++ inference)
        export_model = copy.deepcopy(model)
        export_model.eval()
        export_model.half()
        if torch.cuda.is_available():
            export_model.cuda()
            example_input = torch.zeros(1, 6, 9, 9).half().cuda()
        else:
            example_input = torch.zeros(1, 6, 9, 9).half()
        traced = torch.jit.trace(export_model, example_input)
        traced.save(args.save_model)
        logging.info(f"Exported TorchScript model to {args.save_model}")
        sys.exit(0)

    if not args.load_tree:
        logging.error("--load-tree required for training")
        sys.exit(1)

    train(model, args.load_tree, args.batch_size, args.epochs)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        logging.info(f"Saved model weights to {args.save_model}")


if __name__ == "__main__":
    main()
