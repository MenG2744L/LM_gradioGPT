USE mysql;

CREATE TABLE Login
    (NAME CHAR(30) PRIMARY KEY NOT NULL,
    PASSWORD CHAR(30));

INSERT INTO Login
(NAME, PASSWORD)
 VALUES
('liumeng', '123456'),
('mengliu', '654321'),
('liu', '123'),
('meng', '456');

