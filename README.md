# pi-bot

## Overview
**pi-bot** is a Python-based project, set up for automation and API development with FastAPI and Docker. It’s designed to be modular, testable, and easy to deploy.

## Features
- FastAPI backend for API development
- Dockerized environment for consistent setup
- Pytest for automated testing
- Configurable via environment variables

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/JackronyK/pi-bot.git
   cd pi-bot ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run tests:

   ```bash
   pytest
   ```

## Docker (Optional)

Build and run the Docker container:

```bash
docker-compose up --build
```

## Contributing

* Fork the repository
* Create a new branch
* Make changes and commit
* Open a pull request

## License

[MIT License](LICENSE)

```


