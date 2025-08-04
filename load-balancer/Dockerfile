FROM openjdk:17-jdk-slim
WORKDIR /app
COPY target/load-balancer-1.0.jar app.jar
RUN mkdir -p /app/data
EXPOSE 8082
ENTRYPOINT ["java", "-jar", "app.jar", "--server.port=8082"]

