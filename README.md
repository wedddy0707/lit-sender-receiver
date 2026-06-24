# Lit-Sender-Receiver: PyTorch Lightning Template for Sender-Receiver Games

> [!WARNING]
> **Work in Progress & Personal Template**
>
> This repository is primarily a personal template for my own experiments and is currently under active development.
> As such, interfaces and directory structures may change without prior notice.
> However, anyone is completely free to use, modify, and build upon it for their own projects.

A lightweight template designed for rapid and type-safe research on signaling games (Sender-Receiver games) and emergent communication.
By combining PyTorch Lightning's infrastructure with strict tensor shape checking via jaxtyping, this project aims to maximize the pace of experiments while preventing errors during development.

# Quick Start

```
$ git clone git@github.com:wedddy0707/lit-sender-receiver.git
$ cd lit-sender-receiver
$ uv sync
$ uv run pytest -vv  # if necessary
$ uv run python -m examples.onehot_rnn_reconstruction_game fit --config examples/onehot_rnn_reconstruction_game/example_config.yaml
$ column -s , -t ./lightning_logs/version_0/metrics.csv | less -S
```

# Acknowledgments

This project is significantly inspired by the [EGG (Emergence of lanGuage in Games) toolkit](https://github.com/facebookresearch/EGG) developed by Meta Research (formerly Facebook AI Research).
While this repository is an independent template built to leverage PyTorch Lightning and modern typing, EGG's foundational abstractions for signaling games heavily influenced the architectural design choices made here.

# Citation

While this project is licensed under the 0BSD license and does not require attribution, I would greatly appreciate it if you could cite it if you find it useful in your research:

```bibtex
@misc{lit_sender_receiver,
  author = {Ryo Ueda},
  title = {Lit-Sender-Receiver: PyTorch Lightning Template for Sender-Receiver Games},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/your-username/lit-sender-receiver](https://github.com/your-username/lit-sender-receiver)}}
}
```

## License

This project is licensed under the Zero-Clause BSD (0BSD) License - see the [LICENSE](LICENSE) file for details.