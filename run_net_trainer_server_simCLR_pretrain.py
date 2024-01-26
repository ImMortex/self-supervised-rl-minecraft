from net_trainer_server import main

if __name__ == "__main__":
    server_instance_config: dict = {
        "trainer_name": "simclr_pretrain",
        #"port": "8081"
    }
    main(server_instance_config)  # calls the above defined main function