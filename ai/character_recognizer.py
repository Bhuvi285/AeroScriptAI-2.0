"""
AeroScript 2.0 — AI Character Recognizer
=========================================
PyTorch-based CNN for recognizing hand-drawn air-writing glyphs:
- Digits: 0-9
- Uppercase Letters: A-Z
- Math Operators: +, -, *, /, =, (, )

Features:
- High-efficiency PyTorch CNN (~1.5 MB weights)
- Apple Silicon (MPS) and CPU accelerated
- Auto-trains and caches weights on first launch if not present
- Geometric & template fallback for ultra-reliable recognition
"""

import os
import time
import logging
import numpy as np
import cv2
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

log = logging.getLogger("AeroScript.AI")

# Supported classes (43 total)
CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '+', '-', '*', '/', '=', '(', ')'
]
CHAR_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CLASSES)}
MODEL_PATH  = "models/air_writing_cnn.pth"


if torch is not None:
    class AirWritingCNN(nn.Module):
        """Lightweight 3-stage CNN for 28x28 grayscale air glyphs."""
        def __init__(self, num_classes=len(CLASSES)):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.drop = nn.Dropout(0.3)
            self.fc1 = nn.Linear(128 * 3 * 3, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = self.drop(F.relu(self.fc1(x)))
            return self.fc2(x)
else:
    AirWritingCNN = None


def generate_synthetic_dataset(samples_per_class=350):
    """
    Generates realistic synthetic air-writing stroke samples using diverse
    fonts, scales, thicknesses, rotations, and jitter.
    """
    X = []
    y = []
    
    cv_fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    ]
    
    for label_idx, char in enumerate(CLASSES):
        count = 0
        while count < samples_per_class:
            font = cv_fonts[count % len(cv_fonts)]
            scale = np.random.uniform(0.9, 1.4)
            thick = np.random.choice([1, 2, 3])
            angle = np.random.uniform(-18, 18)
            dx    = np.random.randint(-4, 5)
            dy    = np.random.randint(-4, 5)
            
            canvas = np.zeros((64, 64), dtype=np.uint8)
            (tw, th), bl = cv2.getTextSize(char, font, scale, thick)
            tx = int((64 - tw) / 2 + dx)
            ty = int((64 + th) / 2 + dy)
            
            cv2.putText(canvas, char, (tx, ty), font, scale, 255, thick, cv2.LINE_AA)
            
            # Random rotation
            if abs(angle) > 1.0:
                M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                canvas = cv2.warpAffine(canvas, M, (64, 64), borderValue=0)
                
            # Random morphological variation (dilation/blur)
            if np.random.rand() > 0.5:
                ksize = np.random.choice([2, 3])
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                canvas = cv2.dilate(canvas, kernel, iterations=1)
                
            # Crop to ink bounding box and center into 28x28
            pts = cv2.findNonZero(canvas)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                glyph = canvas[by:by+bh, bx:bx+bw]
                
                # Resize with aspect ratio into 20x20 box
                max_dim = max(bw, bh)
                scale_norm = 20.0 / max(max_dim, 1)
                nw = max(int(bw * scale_norm), 1)
                nh = max(int(bh * scale_norm), 1)
                resized = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA)
                
                # Center in 28x28
                norm_img = np.zeros((28, 28), dtype=np.uint8)
                ox = (28 - nw) // 2
                oy = (28 - nh) // 2
                norm_img[oy:oy+nh, ox:ox+nw] = resized
            else:
                norm_img = cv2.resize(canvas, (28, 28))
                
            # Normalize to [0, 1]
            img_tensor = norm_img.astype(np.float32) / 255.0
            X.append(img_tensor)
            y.append(label_idx)
            count += 1
            
    X = np.array(X, dtype=np.float32)[:, np.newaxis, :, :]  # (N, 1, 28, 28)
    y = np.array(y, dtype=np.int64)
    return X, y


def train_model(model_path=MODEL_PATH, epochs=16, batch_size=128):
    """
    Trains the CNN model on synthetic data and saves weights.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to train the air-writing model")

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Training AirWritingCNN on {device}...")
    
    t0 = time.time()
    X_train, y_train = generate_synthetic_dataset(samples_per_class=350)
    log.info(f"Generated {len(X_train)} synthetic samples in {time.time()-t0:.2f}s")
    
    # Shuffle
    indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]
    
    model = AirWritingCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_y)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
            
        scheduler.step()
        acc = (correct / total) * 100.0
        avg_loss = total_loss / total
        if (epoch + 1) % 4 == 0 or epoch == epochs - 1:
            log.info(f"Epoch {epoch+1:2d}/{epochs} — Loss: {avg_loss:.4f} — Acc: {acc:.1f}%")
            
    torch.save(model.state_dict(), model_path)
    log.info(f"Model saved to {model_path} (Training time: {time.time()-t0:.1f}s)")
    return model


class CharacterClassifier:
    """
    High-level interface for character classification.
    Loads cached weights or triggers fast training on first launch.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CharacterClassifier()
        return cls._instance

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._templates = None
        if torch is None:
            self.device = None
            self._templates = self._build_templates()
            self.model = self._templates
            log.warning("PyTorch is unavailable; using OpenCV template recognition")
            return

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AirWritingCNN().to(self.device)
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                self.model.eval()
                log.info(f"Loaded AirWritingCNN from {self.model_path}")
                return
            except Exception as e:
                log.warning(f"Failed to load weights ({e}). Re-training...")
        
        # Train and save
        self.model = train_model(self.model_path)
        self.model.eval()

    def predict(self, norm_glyph_28x28: np.ndarray) -> tuple[str, float]:
        """
        Classifies a 28x28 normalized grayscale glyph (0..255).
        Returns (predicted_char, confidence_score [0..1]).
        """
        if norm_glyph_28x28 is None or norm_glyph_28x28.size == 0:
            return "?", 0.0

        if torch is None:
            glyph = self._normalize_input(norm_glyph_28x28)
            scores = [self._similarity(glyph, template) for template in self._templates]
            index = int(np.argmax(scores))
            return IDX_TO_CHAR[index], float(max(scores[index], 0.0))
            
        # Ensure 28x28
        if norm_glyph_28x28.shape != (28, 28):
            norm_glyph_28x28 = cv2.resize(norm_glyph_28x28, (28, 28))
            
        tensor = torch.from_numpy(norm_glyph_28x28.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 28, 28)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            
        char = IDX_TO_CHAR[idx.item()]
        return char, float(conf.item())

    def predict_batch(self, glyphs: list[np.ndarray]) -> list[tuple[str, float]]:
        return [self.predict(glyph) for glyph in glyphs]

    def _normalize_input(self, glyph: np.ndarray) -> np.ndarray:
        if glyph.shape != (28, 28):
            glyph = cv2.resize(glyph, (28, 28))
        return glyph.astype(np.float32) / 255.0

    def _build_templates(self) -> list[np.ndarray]:
        templates = []
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
        for char in CLASSES:
            variants = []
            for font in fonts:
                canvas = np.zeros((64, 64), dtype=np.uint8)
                (width, height), _ = cv2.getTextSize(char, font, 1.4, 2)
                origin = ((64 - width) // 2, (64 + height) // 2)
                cv2.putText(canvas, char, origin, font, 1.4, 255, 2, cv2.LINE_AA)
                points = cv2.findNonZero(canvas)
                if points is None:
                    variants.append(np.zeros((28, 28), dtype=np.float32))
                    continue
                x, y, width, height = cv2.boundingRect(points)
                glyph = canvas[y:y + height, x:x + width]
                scale = 20.0 / max(width, height, 1)
                resized = cv2.resize(glyph, (max(int(width * scale), 1), max(int(height * scale), 1)))
                normalized = np.zeros((28, 28), dtype=np.float32)
                offset_x = (28 - resized.shape[1]) // 2
                offset_y = (28 - resized.shape[0]) // 2
                normalized[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized / 255.0
                variants.append(normalized)
            templates.append(np.mean(variants, axis=0))
        return templates

    def _similarity(self, glyph: np.ndarray, template: np.ndarray) -> float:
        glyph_centered = glyph - glyph.mean()
        template_centered = template - template.mean()
        denominator = np.linalg.norm(glyph_centered) * np.linalg.norm(template_centered)
        if denominator == 0:
            return 0.0
        return float(np.sum(glyph_centered * template_centered) / denominator)

    def predict_batch(self, glyphs: list[np.ndarray]) -> list[tuple[str, float]]:
        """
        Batch prediction for multiple 28x28 glyphs.
        """
        if not glyphs:
            return []

        if torch is None:
            return [self.predict(glyph) for glyph in glyphs]
            
        tensors = []
        for g in glyphs:
            if g.shape != (28, 28):
                g = cv2.resize(g, (28, 28))
            tensors.append(g.astype(np.float32) / 255.0)
            
        batch = torch.from_numpy(np.array(tensors)[:, np.newaxis, :, :]).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs  = F.softmax(logits, dim=1)
            confs, indices = torch.max(probs, dim=1)
            
        return [(IDX_TO_CHAR[idx.item()], float(c.item())) for idx, c in zip(indices, confs)]
