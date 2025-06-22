# Manual Setup Tasks

The following items require manual configuration and cannot be automated:

## GitHub Repository Secrets

Set up the following secrets in your GitHub repository settings (Settings > Secrets and variables > Actions):

1. **WANDB_API_KEY**: Your Weights & Biases API key
   - Get from: https://wandb.ai/settings
   - Required for CI/CD pipeline experiment tracking

2. **MLOPS_ENV**: Environment identifier (e.g., "production", "staging")
   - Used for environment-specific configurations

## Weights & Biases Setup

1. **Create W&B Account**: Sign up at https://wandb.ai if you don't have an account
2. **Get API Key**: Go to https://wandb.ai/settings and copy your API key
3. **Update Entity**: Change `WANDB_ENTITY` in config.yaml from "hiroinie" to your W&B username/team
4. **Create Project**: The project "mlops_group8_spotify_prediction" will be created automatically on first run

## Local Development Setup

1. **Environment Variables**: Copy `.env.example` to `.env` and fill in your values:
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

2. **Conda Environment**: Update your conda environment with new dependencies:
   ```bash
   conda env update -f environment.yml
   ```

## Docker Deployment (Optional)

For production deployment:

1. **Build Docker Image**:
   ```bash
   docker build -t mlops-group8-api .
   ```

2. **Run Container**:
   ```bash
   docker run -p 8000:8000 -e WANDB_API_KEY=your_key mlops-group8-api
   ```

## CI/CD Pipeline

The GitHub Actions workflow will automatically:
- Run tests with coverage requirements (≥75%)
- Execute linting checks
- Run the MLflow pipeline
- Require the secrets mentioned above to be configured

## Why These Cannot Be Automated

- **GitHub Secrets**: Require repository admin access and sensitive credentials
- **W&B API Keys**: Personal authentication tokens that cannot be shared
- **Entity Configuration**: Depends on your specific W&B account/organization
- **Environment Variables**: Contain sensitive information that should not be committed to version control

## Next Steps

1. Configure the GitHub secrets listed above
2. Update the W&B entity in config.yaml to match your account
3. Set up local .env file for development
4. Test the pipeline locally before pushing to trigger CI/CD
