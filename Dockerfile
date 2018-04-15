FROM ceruleanwang/quant-base:latest

ENV HARU_DIR   /app/haru
ENV MAYU_DIR   /app/mayu
ENV PYTHONPATH $PYTHONPATH:/app
ENV DEBUG      0

WORKDIR $MAYU_DIR

ENTRYPOINT ["python3.5"]

EXPOSE 80
EXPOSE 9000

