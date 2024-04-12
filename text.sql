USE mysql;

CREATE TABLE Login
    (NAME CHAR(30) PRIMARY KEY NOT NULL,
    PASSWORD CHAR(30),
    AGE INT(5),
    SEX CHAR(5),
    JOB CHAR(20));

INSERT INTO Login
(NAME, PASSWORD)
 VALUES
('liumeng', '123456', 21, '男', '大模型应用实习生'),
('mengliu', '654321', 22, '女', '大模型应用工程师'),
('liu', '123', 32, '男', 'java工程师'),
('meng', '456', 24, '女', 'python工程师');

