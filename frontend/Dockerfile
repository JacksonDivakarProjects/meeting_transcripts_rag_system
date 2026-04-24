FROM node:20-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose the Vite dev server port
EXPOSE 3000

# --host 0.0.0.0 is CRITICAL for Docker, otherwise React only listens on localhost inside the container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]