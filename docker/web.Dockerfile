# Web client container (Nginx + static assets)
FROM nginx:alpine

COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY web/public/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
