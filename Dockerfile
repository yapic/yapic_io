# update the docker image using:
# sudo docker build -t yapic_io-debian .

FROM yapic-debian
RUN pip install --user tifffile munkres

