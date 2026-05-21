# Build stage
FROM node:18-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Pass build args if necessary
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

ARG VITE_STATIC_URL
ENV VITE_STATIC_URL=$VITE_STATIC_URL

RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets to Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Add custom Nginx config to handle React Router history API
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
