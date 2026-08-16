FROM node:22-alpine AS deps
WORKDIR /app
COPY dashboard/package.json /app/package.json
RUN npm install

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules /app/node_modules
COPY dashboard /app
ENV NEXT_PUBLIC_API_BASE=http://api:8101
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production NEXT_PUBLIC_API_BASE=http://api:8101
COPY --from=builder /app/.next /app/.next
COPY --from=builder /app/node_modules /app/node_modules
COPY --from=builder /app/package.json /app/package.json
COPY --from=builder /app/public /app/public
EXPOSE 3101
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD wget -qO- http://127.0.0.1:3101 >/dev/null || exit 1
CMD ["npm", "run", "start"]
