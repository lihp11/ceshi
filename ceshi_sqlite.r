library(DBI)
library(RSQLite)
# 新建或连接已有的磁盘上数据库
con <- dbConnect(RSQLite::SQLite(), 'd:/test.db')
# dbDisconnect(con) # 断开连接
# 临时在内存中建立
# con <- dbConnect(RSQLite::SQLite(), ":memory:")

##########增
# 导入已有的数据框, 参数为“连接对象， 表名，数据框”
dbWriteTable(con, 'mtcar', mtcars )
# 建立新表
dbExecute(con, 'CREATE TABLE test (name char(10), age int(10))')
dbExecute(con, 'INSERT INTO test (name, age) VALUES ("Tom", 25), ("Jelly",26)')
# 数据量过大，可以批次插入
rs <- dbSendStatement(con, 'INSERT INTO test (name, age) VALUES ("Tom", 25), ("Jelly",26)' )
dbHasCompleted(rs) # 判断操作是否完成
dbClearResult(rs) # 清空内存


##########查
# 方法1
y <- dbGetQuery(con, 'SELECT * FROM test')	# dataframe
# 方法2
con <- dbConnect(RSQLite::SQLite(), ":memory:")
dbWriteTable(con, "mtcars", mtcars)
rs <- dbSendQuery(con, "SELECT * FROM mtcars WHERE cyl = 4;")
dbFetch(rs, n=10) # 获取10行
dbFetch(rs, n=10) # 再获取10行
dbClearResult(rs) # 清空rs句柄中的内容
dbDisconnect(con)

##########删
# 删除记录前，要先查询
dbGetQuery(con, 'SELECT * FROM mtcar WHERE mpg > 30')
# 删除记录
dbExecute(con, 'DELETE FROM mtcar WHERE mpg > 30')
# 删除所有记录，谨慎操作
dbExecute(con, 'DROP TABLE test')
dbRemoveTable(con, 'test')

##########改
# 修改记录前也是先查找
dbGetQuery(con, 'SELECT * FROM mtcar WHERE am ==1')
# 然后我们将mpg=21,且qsec=16.46 记录 的mpg修改为21.1
dbExecute(con, 'UPDATE mtcar SET mpg = 21.1 WHERE mpg =21.0 AND qsec =16.46')
# 事后验证
dbGetQuery(con, 'SELECT * FROM mtcar WHERE mpg = 21.1')




############后悔
con <- dbConnect(RSQLite::SQLite(), ":memory:")
dbWriteTable(con, "cash", data.frame(amount = 100))
# 测试dbCommit
dbBegin(con)
withdrawal <- 300
dbExecute(con, "UPDATE cash SET amount = amount + ?", list(withdrawal))
dbCommit(con)
# 测试回滚
dbBegin(con)
withdrawl <- 5000
dbExecute(con, "UPDATE cash SET amount = amount + ?", list(withdrawal))
## 查看结果
dbGetQuery(con,'SELECT * FROM cash')
## 回滚
dbRollback(con)
## 检查结果
dbGetQuery(con,'SELECT * FROM cash')
dbDisconnect(con)


# dbWriteTable(连接对象, 表名, 数据框): 将数据框保存为表
# dbReadTable(连接对象, 表名 ): 读取指定表中所有内容
# dbListTable(连接对象)： 列出数据库中包含的表格
# dbListFields(连接对象, 表名)： 列出制定表格列名
# dbReadTable(连接对象, 表名): 将指定表读取为数据框
