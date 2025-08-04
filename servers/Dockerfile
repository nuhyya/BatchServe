FROM openjdk:17-jdk-slim
WORKDIR /app
COPY target/model-server-1.0.jar app.jar
RUN mkdir -p /app/data
ENV SERVER_PORT=8084
EXPOSE ${SERVER_PORT}
ENTRYPOINT ["sh", "-c", "java -Dserver.port=${SERVER_PORT} -jar app.jar"]
