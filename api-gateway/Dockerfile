FROM openjdk:17-jdk-slim
WORKDIR /app
COPY target/api-gateway-1.0.jar app.jar
RUN mkdir -p /app/data
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]

