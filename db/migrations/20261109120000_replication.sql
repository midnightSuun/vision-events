-- migrate:up

ALTER USER vision WITH REPLICATION;

-- migrate:down

ALTER USER vision WITH NOREPLICATION;
