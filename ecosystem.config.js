module.exports = {
  apps: [
    {
      name: "Snap Jobs Module api",
      script: "main.py",
      interpreter: "python3",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
        ENVIRONMENT: "production",
        DB_HOST: "localhost",
        DB_USER: "your_db_user",
        DB_PASSWORD: "your_db_password",
        DB_NAME: "sjm_db",
        CLAUDE_API_KEY: "your_claude_api_key",
        OPENAI_API_KEY: "your_openai_api_key",
        ENCRYPTION_KEY: "your_base64_encryption_key",
      },
    },
  ],
};
