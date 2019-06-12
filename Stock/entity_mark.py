# -*- coding: utf-8 -*-
from sqlalchemy import Column, String, Integer, ForeignKey, Date, Text, Boolean
from database import Base, db_session
class EntityMark(Base):

    #表名
    __tablename__ ='entity_mark'
    #表结构
    id = Column(Integer,autoincrement=True, primary_key=True)
    content = Column(Text, nullable=False)
    passed = Column(Boolean, default=False)  # 是否审核通过
    reviewed = Column(Boolean, default=False)  # 是否被审核过
    ver_date = Column(Date, nullable=True)
    user_id = Column(Integer, nullable=True)  # 是审核数据的人的id！不是数据上传人
    # 查询构造器、、、
    query = db_session.query_property()

    def __init__(self, content):
        self.content = content

    def __repr__(self):
        entity_dict = {
            u"content": self.content
        }
        return str(entity_dict)

    def __dir__(self):
        entity_dict = {
            u"content": self.content
        }
        return str(entity_dict)
