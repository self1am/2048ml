const boardSize = 4;
let board = [];
let score = 0;

document.addEventListener("DOMContentLoaded", () => {
  createBoard();
  addTile();
  addTile();
  updateBoard();
});

document.addEventListener("keydown", handleKey);

function createBoard() {
  board = Array(boardSize).fill().map(() => Array(boardSize).fill(0));
}

function updateBoard() {
  const gameBoard = document.getElementById("game-board");
  gameBoard.innerHTML = "";
  board.forEach(row => {
    row.forEach(value => {
      const tile = document.createElement("div");
      tile.className = "tile";
      if (value > 0) {
        tile.textContent = value;
        tile.dataset.value = value;
      }
      gameBoard.appendChild(tile);
    });
  });
  document.getElementById("score").textContent = "Score: " + score;
}

function addTile() {
  let emptyTiles = [];
  for (let i = 0; i < boardSize; i++) {
    for (let j = 0; j < boardSize; j++) {
      if (board[i][j] === 0) emptyTiles.push({ x: i, y: j });
    }
  }
  if (emptyTiles.length === 0) return;
  const { x, y } = emptyTiles[Math.floor(Math.random() * emptyTiles.length)];
  board[x][y] = Math.random() > 0.1 ? 2 : 4;
}

function handleKey(e) {
  switch (e.key) {
    case "ArrowUp":
      moveUp();
      break;
    case "ArrowDown":
      moveDown();
      break;
    case "ArrowLeft":
      moveLeft();
      break;
    case "ArrowRight":
      moveRight();
      break;
    default:
      return;
  }
  addTile();
  updateBoard();
  if (isGameOver()) alert("Game Over!");
}

function moveUp() {
  for (let j = 0; j < boardSize; j++) {
    let compressed = [];
    for (let i = 0; i < boardSize; i++) {
      if (board[i][j] !== 0) compressed.push(board[i][j]);
    }
    merge(compressed);
    for (let i = 0; i < boardSize; i++) {
      board[i][j] = compressed[i] || 0;
    }
  }
}

function moveDown() {
  for (let j = 0; j < boardSize; j++) {
    let compressed = [];
    for (let i = boardSize - 1; i >= 0; i--) {
      if (board[i][j] !== 0) compressed.push(board[i][j]);
    }
    merge(compressed);
    for (let i = 0; i < boardSize; i++) {
      board[boardSize - i - 1][j] = compressed[i] || 0;
    }
  }
}

function moveLeft() {
  for (let i = 0; i < boardSize; i++) {
    let compressed = board[i].filter(v => v !== 0);
    merge(compressed);
    board[i] = [...compressed, ...Array(boardSize - compressed.length).fill(0)];
  }
}

function moveRight() {
  for (let i = 0; i < boardSize; i++) {
    let compressed = board[i].filter(v => v !== 0);
    merge(compressed);
    board[i] = [...Array(boardSize - compressed.length).fill(0), ...compressed];
  }
}

function merge(arr) {
  for (let i = 0; i < arr.length - 1; i++) {
    if (arr[i] === arr[i + 1]) {
      arr[i] *= 2;
      score += arr[i];
      arr.splice(i + 1, 1);
    }
  }
}

function isGameOver() {
  for (let i = 0; i < boardSize; i++) {
    for (let j = 0; j < boardSize; j++) {
      if (board[i][j] === 0) return false;
      if (i < boardSize - 1 && board[i][j] === board[i + 1][j]) return false;
      if (j < boardSize - 1 && board[i][j] === board[i][j + 1]) return false;
    }
  }
  return true;
}

function resetGame() {
  createBoard();
  score = 0;
  addTile();
  addTile();
  updateBoard();
}
