# 🤖 RoboStack - A template for industry-grade research code

[![arXiv](https://img.shields.io/badge/arXiv-2508.10480-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://www.arxiv.org/abs/2508.10480)
[![GitHub stars](https://img.shields.io/github/stars/antonioterpin/robostack?style=social)](https://github.com/antonioterpin/robostack/stargazers)
[![codecov](https://codecov.io/gh/antonioterpin/robostack/graph/badge.svg?token=J49B8TFDSM)](https://codecov.io/gh/antonioterpin/robostack)
[![Tests](https://github.com/antonioterpin/robostack/actions/workflows/test.yaml/badge.svg)](https://github.com/antonioterpin/robostack/actions/workflows/test.yaml)
[![Code Style](https://github.com/antonioterpin/robostack/actions/workflows/code-style.yaml/badge.svg)](https://github.com/antonioterpin/robostack/actions/workflows/code-style.yaml)
[![PyPI version](https://img.shields.io/pypi/v/pinet-hcnn.svg)](https://pypi.org/project/pinet-hcnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Ready to build something amazing?** Fork this template and start your next research project with a solid foundation! 🚀

The works using this template typically follow this structure for the README. The above paragraph would be substituted by a short description of the work.

## ✨ Features
I recommend introducing a bullet points of features whenever applicable.
- 🚀 **Modern Python tooling** with [uv](https://github.com/astral-sh/uv) for fast dependency management.
- 🧪 **Test-driven development** with pytest and comprehensive coverage reporting.
- 🔧 **Automated code quality** with pre-commit hooks, black, isort, and flake8.
- 📝 **Documentation-first** approach with structured contributing guidelines.
- 🐳 **Docker support** for reproducible environments
- ⚡ **GitHub Actions** for CI/CD with parallel testing and style checks.
- 🎯 **JAX-ready** configuration for high-performance ML research.
- 📊 **Logging integration** with [goggles](https://github.com/antonioterpin/goggles).

## 🏗️ Projects Built with RoboStack
I think it is important to prove the adoption of your software, and this can be done by highlighting projects that did so. This template has been battle-tested across multiple research projects:

[![FluidSControl](https://img.shields.io/badge/GitHub-antonioterpin%2Ffluidscontrol-2ea44f?logo=github)](https://github.com/antonioterpin/fluidscontrol)
[![FlowGym](https://img.shields.io/badge/GitHub-antonioterpin%2Fflowgym-2ea44f?logo=github)](https://github.com/antonioterpin/flowgym)
[![SynthPix](https://img.shields.io/badge/GitHub-antonioterpin%2Fsynthpix-2ea44f?logo=github)](https://github.com/antonioterpin/synthpix)
[![Πnet](https://img.shields.io/badge/GitHub-antonioterpin%2Fpinet-2ea44f?logo=github)](https://github.com/antonioterpin/pinet)
[![Glitch](https://img.shields.io/badge/GitHub-antonioterpin%2Fglitch-2ea44f?logo=github)](https://github.com/antonioterpin/glitch)
[![Goggles](https://img.shields.io/badge/GitHub-antonioterpin%2Fgoggles-2ea44f?logo=github)](https://github.com/antonioterpin/goggles)

## 🚀 Quick Start

### Installation
Explain how to install the repo, and its prerequisites. If it could be used by other repos, should be deployed on [Pypi](https://pypi.org/) and installed with `uv add my_package` or `pip install my_package`. Otherwise, you can copy the description for development, which is described in the [Contributing Guide](./CONTRIBUTING.md).

### Supported Platforms 💻

| Platform | CPU | GPU (NVIDIA) | GPU (AMD) | GPU (Intel) | Apple Silicon |
|----------|-----|-------------|-----------|-------------|---------------|
| Linux    | ✅   | ✅          | ❌         | ❌          | n/a           |
| macOS    | ✅   | n/a         | n/a       | n/a         | ✅            |
| Windows  | ✅   | ✅          | ❌         | ❌          | n/a           |

## 🔥 Results
Explain how to reproduce the results, with a few copy-paste commands and/or examples.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information on:
- Development workflow and branch management
- Code style requirements and automated checks
- Testing standards and coverage expectations
- PR preparation and commit message conventions
I postponed writing this template for quite too long. Over time I may have reviewed a tons of PRs and gave feedback on the SWE. If you believe some of this feedback was rather unconventional but good, please open a PR to this template to edit [Contributing Guide](./CONTRIBUTING.md).. Please do the same if you think some rules should be amended or we should introduce new ones 🙏.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
