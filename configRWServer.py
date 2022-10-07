import configparser


def configfile_reader(file_path):
    # 读取配置文件
    config_reader = configparser.ConfigParser()
    config_reader.read(file_path)

    # 获取所有的section,返回是list
    read_sections = config_reader.sections()

    read_list = []

    for section_name in read_sections:

        # 获取所有的属性中的键值对
        read_items = config_reader.items(section_name)

        # 构造读取列表
        for _ in read_items:

            read_list.append((section_name, _[0], _[1]))

    return read_list


def configfile_reader_sections(file_path):
    # 读取配置文件
    config_reader = configparser.ConfigParser()
    config_reader.read(file_path)

    # 获取所有的section,返回是list
    read_sections = config_reader.sections()

    return read_sections


def configfile_reader_option(file_path):
    # 读取配置文件
    config_reader = configparser.ConfigParser()
    config_reader.read(file_path)

    # 获取所有的section,返回是list
    read_sections = config_reader.sections()

    read_list = []

    for section_name in read_sections:

        # 获取所有的属性名
        read_option = config_reader.options(section_name)

        # 构造读取列表
        for _ in read_option:
            read_list.append((section_name, _))

    return read_list


def cofigfile_reader_value(file_path, section_name, option_name):
    # 读取配置文件
    config_reader = configparser.ConfigParser()
    config_reader.read(file_path)

    # 获取配置信息中特定的取值
    return config_reader.getint(section_name, option_name)


def configfile_write(file_path, section_add_list):
    # 写入配置文件,如果没有则新建文件，有则添加
    config_writer = configparser.ConfigParser()
    config_writer.read(file_path)

    # 获取所有需要添加的section_name
    section_name_list = [section_add_list[_][0]
                         for _ in range(len(section_add_list))]

    section_name_deduplication = list(set(section_name_list))

    # 添加不重复的section
    for section_name_add in section_name_deduplication:

        if not config_writer.has_section(section_name_add):

            config_writer.add_section(section_name_add)

    # 添加section中的各项信息
    for section_content in section_add_list:

        section_name, option_name, value_value = section_content[
            0], section_content[1], section_content[2]

        config_writer.set(section_name, option_name, value_value)

    # 写入到配置文件中
    config_writer.write(open(file_path, 'w'))


def configfile_revise(file_path, section_revise_list):
    # 修改配置文件
    config_writer = configparser.ConfigParser()
    config_writer.read(file_path)

    for _ in section_revise_list:

        if config_writer.has_option(_[0], _[1]):

            # 对已有的配置信息做修改
            config_writer.remove_option(_[0], _[1])
            config_writer.set(_[0], _[1], _[2])

        elif not config_writer.has_section(_[0]):

            # 添加没有的配置信息
            config_writer.add_section(_[0])
            config_writer.set(_[0], _[1], _[2])

        else:

            # 添加没有的配置信息
            config_writer.set(_[0], _[1], _[2])

    # 写入到配置文件中
    config_writer.write(open(file_path, 'w'))


def configfile_delete(file_path, section_delete_list):
    # 修改配置文件
    config_writer = configparser.ConfigParser()
    config_writer.read(file_path)

    for _ in section_delete_list:

        if config_writer.has_option(_[0], _[1]):

            # 对已有的配置信息做删除
            config_writer.remove_option(_[0], _[1])

        else:

            # 删除失败
            print('failed to delete!')

    # 写入到配置文件中
    config_writer.write(open(file_path, 'w'))


if __name__ == '__main__':

    configFileName = './test1.conf'
    # 读取所有的配置信息
    read_list = configfile_reader(configFileName)

    print(read_list)

    # 读取所有的section名称
    read_sections = configfile_reader_sections(configFileName)

    print(read_sections)

    # 读取所有的section和option的键值对
    read_options = configfile_reader_option(configFileName)

    print(read_options)

    # 读取特定的配置属性的值
    read_value = cofigfile_reader_value(
        configFileName, 'edgeDetection', 'minThreshold')

    print(type(read_value), read_value)

    # # 添加配置信息
    # section_add_list = [('ip', 'name', 'xiaoli'),
    #                     ('ip', 'ip', '1.1.1.1'), ('adress', 'path', './c/')]
    # configfile_write('./test2.conf', section_add_list)

    # 修改配置信息
    section_revise_list = [('edgeDetection', 'minThreshold', '60')]

    configfile_revise(configFileName, section_revise_list)

    # # 删除某一条配置信息
    # section_delete_list = [('phone', 'path', '123')]
    # configfile_delete('./test2.conf', section_delete_list)
