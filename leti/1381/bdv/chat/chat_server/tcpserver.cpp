#include "tcpserver.h"

TcpServer::TcpServer(int max_threads, QObject *parent) : QTcpServer(parent)
{
    pool = new QThreadPool(0);
    pool->setMaxThreadCount(max_threads);
}

TcpServer::~TcpServer()
{
    for(int i = 0; i < sockets.size(); i++)
    {
        QTcpSocket* s = std::get<1>(sockets[i]);
        QMutex* m = std::get<2>(sockets[i]);
        delete s;
        delete m;
    }
    std::cout << "Sever finished" << std::endl;
    delete[] pool;
}

void TcpServer::incomingConnection(qintptr socketDescriptor)
{
    QTcpSocket* socket = new QTcpSocket();
    QMutex* mutex = new QMutex();
    std::tuple<std::string, QTcpSocket*, QMutex*> elem;
    std::get<1>(elem) = socket;
    std::get<2>(elem) = mutex;
    sockets.push_back(elem);
    std::cout << "Incoming connection " << std::endl;
    socket->setSocketDescriptor(socketDescriptor);
    connect(socket, SIGNAL(readyRead()), this, SLOT(ready_read()));
}

void TcpServer::start(int portnum)
{
    port = portnum;
    if(listen(QHostAddress::Any, port))
    {
        std::cout << "Server started." << std::endl;
    }
    else
    {
        std::cout << "Starting server error." << std::endl;
    }
}

void TcpServer::ready_read()
{
    QTcpSocket* socket = (QTcpSocket*)QObject::sender();
    ReadAndHandle* mytask = new ReadAndHandle(socket, sockets);
    mytask->setAutoDelete(true);

    connect(mytask, SIGNAL(setting_username(QTcpSocket*,char*,int)), this, SLOT(set_username(QTcpSocket*,char*,int)));
    connect(mytask, SIGNAL(deleting_socket_from_store(QTcpSocket*)), this, SLOT(close_socket(QTcpSocket*)));
    pool->start(mytask);
}


void TcpServer::close_socket(QTcpSocket* socket)
{
    QMutex* m;
    socket->close();
    delete_socket_from_store(socket, &m);
    delete socket;
    delete m;
}

void TcpServer::delete_socket_from_store(QTcpSocket* sock, QMutex** mut)
{
    for(auto it = sockets.begin(); it != sockets.end(); it++)
    {
        if(std::get<1>(*it) == sock)
        {
            mut[0] = std::get<2>(*it);
            sockets.erase(it);
            break;
        }
    }
}

void TcpServer::set_username(QTcpSocket* socket, char* name, int size)
{
    std::string s(name, size);
    delete[] name;

    for(auto it = sockets.begin(); it != sockets.end(); it++)
    {
        if(std::get<1>(*it) == socket)
        {
            std::get<0>(*it) = s;
            break;
        }
    }
}
