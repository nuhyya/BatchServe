FROM openjdk:17-jdk-slim
WORKDIR /app
COPY target/server-manager-1.0.jar app.jar
RUN mkdir -p /app/data
EXPOSE 8083
ENTRYPOINT ["java", "-jar", "app.jar"]
