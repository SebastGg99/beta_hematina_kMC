"""Árbol de Fenwick (binary indexed tree) de pesos float, en Python puro.

Guarda el peso total de cada clase de eventos (n_c · r_c) y permite, en O(log K):
- sumar un incremento al peso de una clase (`add`);
- elegir la clase cuya suma acumulada contiene un número dado (`find`).

Las sumas y restas repetidas de pesos muy distintos (las tasas pueden ir de e^-20 a
e^+8) acumulan error de redondeo, así que el motor llama a `rebuild` de vez en cuando
con los pesos exactos.
"""

from typing import List, Sequence


class Fenwick:
    """Pesos no negativos w_0..w_{n-1} con sumas prefijas en O(log n)."""

    def __init__(self, capacity: int = 64):
        self.n = max(1, int(capacity))
        self.tree: List[float] = [0.0] * (self.n + 1)   # índice 1-based
        self.total = 0.0
        self._top = 1 << (self.n.bit_length() - 1)     # mayor potencia de 2 <= n

    def add(self, i: int, delta: float) -> None:
        """w_i += delta."""
        self.total += delta
        tree, n = self.tree, self.n
        i += 1
        while i <= n:
            tree[i] += delta
            i += i & (-i)

    def find(self, target: float) -> int:
        """Menor i con w_0 + ... + w_i > target (clases de peso 0 se saltan).

        Si `target` >= total (por redondeo) devuelve n − 1; el llamador comprueba
        que la clase elegida no esté vacía.
        """
        tree, n = self.tree, self.n
        pos = 0
        rem = target
        step = self._top
        while step:
            nxt = pos + step
            if nxt <= n and tree[nxt] <= rem:
                pos = nxt
                rem -= tree[nxt]
            step >>= 1
        return pos if pos < n else n - 1

    def rebuild(self, weights: Sequence[float], capacity: int = 0) -> None:
        """Reconstruye desde pesos exactos en O(n) (y amplía la capacidad si se pide)."""
        n = max(self.n, int(capacity), len(weights))
        tree = [0.0] * (n + 1)
        for i, w in enumerate(weights):
            tree[i + 1] = float(w)
        for i in range(1, n + 1):          # construcción lineal estándar
            j = i + (i & (-i))
            if j <= n:
                tree[j] += tree[i]
        self.n = n
        self.tree = tree
        self._top = 1 << (n.bit_length() - 1)
        self.total = float(sum(float(w) for w in weights))

    def prefix(self, i: int) -> float:
        """w_0 + ... + w_i (para tests)."""
        s = 0.0
        i += 1
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s
