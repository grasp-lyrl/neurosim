// Copyright Neurosim contributors. Licensed under the repo LICENSE.

#include "JsonSocketServer.h"
#include "FloatingCameraPawn.h"

#include "Async/Async.h"
#include "Dom/JsonObject.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

#include <arpa/inet.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

DEFINE_LOG_CATEGORY_STATIC(LogNeurosimSocket, Log, All);

static constexpr const char* kServerVersion = "0.1.0";

FJsonSocketServer::FJsonSocketServer() = default;

FJsonSocketServer::~FJsonSocketServer()
{
    Stop();
}

bool FJsonSocketServer::Start(const FString& SocketPath)
{
    BoundPath = SocketPath;

    ListenFd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (ListenFd < 0)
    {
        UE_LOG(LogNeurosimSocket, Error, TEXT("socket() failed: %s"), UTF8_TO_TCHAR(strerror(errno)));
        return false;
    }

    const FTCHARToUTF8 Utf8Path(*SocketPath);
    if (Utf8Path.Length() >= (int32)sizeof(sockaddr_un::sun_path))
    {
        UE_LOG(LogNeurosimSocket, Error, TEXT("socket path too long: %s"), *SocketPath);
        close(ListenFd);
        ListenFd = -1;
        return false;
    }

    // Best-effort cleanup of a stale socket file.
    unlink(Utf8Path.Get());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    FCStringAnsi::Strncpy(addr.sun_path, Utf8Path.Get(), sizeof(addr.sun_path) - 1);

    if (bind(ListenFd, (sockaddr*)&addr, sizeof(addr)) < 0)
    {
        UE_LOG(LogNeurosimSocket, Error, TEXT("bind(%s) failed: %s"),
               *SocketPath, UTF8_TO_TCHAR(strerror(errno)));
        close(ListenFd);
        ListenFd = -1;
        return false;
    }

    if (listen(ListenFd, 1) < 0)
    {
        UE_LOG(LogNeurosimSocket, Error, TEXT("listen() failed: %s"), UTF8_TO_TCHAR(strerror(errno)));
        close(ListenFd);
        ListenFd = -1;
        return false;
    }

    Thread.Reset(FRunnableThread::Create(this, TEXT("NeurosimSocketServer")));
    return Thread.IsValid();
}

void FJsonSocketServer::Stop()
{
    bStopRequested.store(true);
    if (ListenFd >= 0)
    {
        shutdown(ListenFd, SHUT_RDWR);
        close(ListenFd);
        ListenFd = -1;
    }
    if (Thread.IsValid())
    {
        Thread->WaitForCompletion();
        Thread.Reset();
    }
    if (!BoundPath.IsEmpty())
    {
        const FTCHARToUTF8 Utf8Path(*BoundPath);
        unlink(Utf8Path.Get());
        BoundPath.Empty();
    }
}

uint32 FJsonSocketServer::Run()
{
    while (!bStopRequested.load())
    {
        sockaddr_un peer{};
        socklen_t peerLen = sizeof(peer);
        const int ClientFd = accept(ListenFd, (sockaddr*)&peer, &peerLen);
        if (ClientFd < 0)
        {
            if (bStopRequested.load()) break;
            UE_LOG(LogNeurosimSocket, Warning, TEXT("accept() failed: %s"), UTF8_TO_TCHAR(strerror(errno)));
            continue;
        }
        UE_LOG(LogNeurosimSocket, Log, TEXT("Client connected"));
        ServeClient(ClientFd);
        close(ClientFd);
        UE_LOG(LogNeurosimSocket, Log, TEXT("Client disconnected"));
    }
    return 0;
}

bool FJsonSocketServer::ReadFrame(int ClientFd, TArray<uint8>& OutBytes)
{
    uint32 LenBE = 0;
    uint8* LenPtr = (uint8*)&LenBE;
    int32 Got = 0;
    while (Got < 4)
    {
        const ssize_t n = recv(ClientFd, LenPtr + Got, 4 - Got, 0);
        if (n <= 0) return false;
        Got += (int32)n;
    }
    const uint32 Len = ntohl(LenBE);
    if (Len == 0 || Len > (16u << 20)) return false; // 16 MiB hard cap
    OutBytes.SetNumUninitialized((int32)Len);
    int32 Read = 0;
    while (Read < (int32)Len)
    {
        const ssize_t n = recv(ClientFd, OutBytes.GetData() + Read, (int32)Len - Read, 0);
        if (n <= 0) return false;
        Read += (int32)n;
    }
    return true;
}

bool FJsonSocketServer::WriteFrame(int ClientFd, const TArray<uint8>& Bytes)
{
    const uint32 LenBE = htonl((uint32)Bytes.Num());
    const uint8* LenPtr = (const uint8*)&LenBE;
    int32 Sent = 0;
    while (Sent < 4)
    {
        const ssize_t n = send(ClientFd, LenPtr + Sent, 4 - Sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        Sent += (int32)n;
    }
    Sent = 0;
    while (Sent < Bytes.Num())
    {
        const ssize_t n = send(ClientFd, Bytes.GetData() + Sent, Bytes.Num() - Sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        Sent += (int32)n;
    }
    return true;
}

void FJsonSocketServer::ServeClient(int ClientFd)
{
    TArray<uint8> InBytes;
    while (!bStopRequested.load() && ReadFrame(ClientFd, InBytes))
    {
        const FUTF8ToTCHAR InUtf8((const ANSICHAR*)InBytes.GetData(), InBytes.Num());
        const FString InStr(InUtf8.Length(), InUtf8.Get());

        TSharedPtr<FJsonObject> Request;
        const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(InStr);
        TSharedPtr<FJsonObject> Response = MakeShared<FJsonObject>();
        if (!FJsonSerializer::Deserialize(Reader, Request) || !Request.IsValid())
        {
            Response->SetNumberField(TEXT("id"), -1);
            Response->SetStringField(TEXT("error"), TEXT("malformed json"));
        }
        else
        {
            Response = Dispatch(Request);
        }

        FString OutStr;
        const TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
            TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&OutStr);
        FJsonSerializer::Serialize(Response.ToSharedRef(), Writer);

        const FTCHARToUTF8 OutUtf8(*OutStr);
        TArray<uint8> OutBytes;
        OutBytes.Append((const uint8*)OutUtf8.Get(), OutUtf8.Length());
        if (!WriteFrame(ClientFd, OutBytes)) break;
    }
}

TSharedPtr<FJsonObject> FJsonSocketServer::Dispatch(const TSharedPtr<FJsonObject>& Request)
{
    TSharedPtr<FJsonObject> Response = MakeShared<FJsonObject>();
    double Id = -1;
    Request->TryGetNumberField(TEXT("id"), Id);
    Response->SetNumberField(TEXT("id"), Id);

    FString Method;
    if (!Request->TryGetStringField(TEXT("method"), Method))
    {
        Response->SetStringField(TEXT("error"), TEXT("missing method"));
        return Response;
    }

    const TSharedPtr<FJsonObject> Params = Request->HasField(TEXT("params"))
        ? Request->GetObjectField(TEXT("params")) : MakeShared<FJsonObject>();

    FString Err;
    TSharedPtr<FJsonObject> Result;
    if      (Method == TEXT("handshake"))      Result = HandleHandshake(Params, Err);
    else if (Method == TEXT("set_agent_pose")) Result = HandleSetAgentPose(Params, Err);
    else if (Method == TEXT("get_agent_pose")) Result = HandleGetAgentPose(Params, Err);
    else if (Method == TEXT("open_level"))     Result = HandleOpenLevel(Params, Err);
    else if (Method == TEXT("render_frame"))   Result = HandleRenderFrame(Params, Err);
    else if (Method == TEXT("shutdown"))       Result = HandleShutdown(Params, Err);
    else Err = FString::Printf(TEXT("unknown method: %s"), *Method);

    if (!Err.IsEmpty() || !Result.IsValid())
    {
        Response->SetStringField(TEXT("error"),
                                 Err.IsEmpty() ? TEXT("handler returned null") : *Err);
    }
    else
    {
        Response->SetObjectField(TEXT("result"), Result);
    }
    return Response;
}

// ---------------------------------------------------------------------------
// Handlers
//
// All handlers that touch UWorld / AActor must run on the game thread. We use
// FFunctionGraphTask + FEvent to marshal + wait.

template <typename F>
static void RunOnGameThreadAndWait(F&& Fn)
{
    if (IsInGameThread()) { Fn(); return; }
    FEvent* Done = FPlatformProcess::GetSynchEventFromPool(false);
    AsyncTask(ENamedThreads::GameThread, [&]() { Fn(); Done->Trigger(); });
    Done->Wait();
    FPlatformProcess::ReturnSynchEventToPool(Done);
}

TSharedPtr<FJsonObject> FJsonSocketServer::HandleHandshake(
    const TSharedPtr<FJsonObject>& Params, FString& OutError)
{
    TSharedPtr<FJsonObject> R = MakeShared<FJsonObject>();
    R->SetStringField(TEXT("server_version"), UTF8_TO_TCHAR(kServerVersion));
    R->SetNumberField(TEXT("pid"), (double)FPlatformProcess::GetCurrentProcessId());
    // Phase 2 will populate sensors {name -> {w,h,format,fd_index}} and attach fds
    // via SCM_RIGHTS. For Phase 1 we advertise no sensors.
    R->SetObjectField(TEXT("sensors"), MakeShared<FJsonObject>());
    return R;
}

TSharedPtr<FJsonObject> FJsonSocketServer::HandleSetAgentPose(
    const TSharedPtr<FJsonObject>& Params, FString& OutError)
{
    const TArray<TSharedPtr<FJsonValue>>* PosArr = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* RotArr = nullptr;
    if (!Params->TryGetArrayField(TEXT("position"), PosArr) || PosArr->Num() != 3 ||
        !Params->TryGetArrayField(TEXT("rotation"), RotArr) || RotArr->Num() != 4)
    {
        OutError = TEXT("set_agent_pose expects position[3], rotation[4]");
        return nullptr;
    }
    const FVector PosHabitat(
        (*PosArr)[0]->AsNumber(), (*PosArr)[1]->AsNumber(), (*PosArr)[2]->AsNumber());
    // Habitat quaternion order is [w, x, y, z].
    const FQuat QuatHabitat(
        (*RotArr)[1]->AsNumber(), (*RotArr)[2]->AsNumber(),
        (*RotArr)[3]->AsNumber(), (*RotArr)[0]->AsNumber());

    bool bOk = false;
    RunOnGameThreadAndWait([&]() {
        if (UWorld* World = GEngine ? GEngine->GetCurrentPlayWorld() : nullptr)
        {
            AFloatingCameraPawn* Pawn = AFloatingCameraPawn::GetOrSpawn(World);
            if (Pawn) { Pawn->SetHabitatPose(PosHabitat, QuatHabitat); bOk = true; }
        }
    });
    if (!bOk) { OutError = TEXT("no active world / pawn"); return nullptr; }
    TSharedPtr<FJsonObject> R = MakeShared<FJsonObject>();
    R->SetBoolField(TEXT("ok"), true);
    return R;
}

TSharedPtr<FJsonObject> FJsonSocketServer::HandleGetAgentPose(
    const TSharedPtr<FJsonObject>& Params, FString& OutError)
{
    FVector Pos(0);
    FQuat Quat = FQuat::Identity;
    bool bOk = false;
    RunOnGameThreadAndWait([&]() {
        if (UWorld* World = GEngine ? GEngine->GetCurrentPlayWorld() : nullptr)
        {
            AFloatingCameraPawn* Pawn = AFloatingCameraPawn::GetOrSpawn(World);
            if (Pawn) { Pawn->GetHabitatPose(Pos, Quat); bOk = true; }
        }
    });
    if (!bOk) { OutError = TEXT("no active world / pawn"); return nullptr; }

    TSharedPtr<FJsonObject> R = MakeShared<FJsonObject>();
    TArray<TSharedPtr<FJsonValue>> PosArr = {
        MakeShared<FJsonValueNumber>(Pos.X),
        MakeShared<FJsonValueNumber>(Pos.Y),
        MakeShared<FJsonValueNumber>(Pos.Z) };
    TArray<TSharedPtr<FJsonValue>> RotArr = {
        MakeShared<FJsonValueNumber>(Quat.W),
        MakeShared<FJsonValueNumber>(Quat.X),
        MakeShared<FJsonValueNumber>(Quat.Y),
        MakeShared<FJsonValueNumber>(Quat.Z) };
    R->SetArrayField(TEXT("position"), PosArr);
    R->SetArrayField(TEXT("rotation"), RotArr);
    return R;
}

TSharedPtr<FJsonObject> FJsonSocketServer::HandleOpenLevel(
    const TSharedPtr<FJsonObject>& Params, FString& OutError)
{
    FString LevelPath;
    if (!Params->TryGetStringField(TEXT("level_path"), LevelPath))
    {
        OutError = TEXT("open_level expects level_path");
        return nullptr;
    }
    RunOnGameThreadAndWait([&]() {
        if (UWorld* World = GEngine ? GEngine->GetCurrentPlayWorld() : nullptr)
        {
            UGameplayStatics::OpenLevel(World, FName(*LevelPath));
        }
    });
    TSharedPtr<FJsonObject> R = MakeShared<FJsonObject>();
    R->SetBoolField(TEXT("ok"), true);
    return R;
}

TSharedPtr<FJsonObject> FJsonSocketServer::HandleRenderFrame(
    const TSharedPtr<FJsonObject>& Params, FString& OutError)
{
    // Phase 1: no render targets yet; just round-trip so the client state machine
    // can be tested. Phase 2 will kick SceneCapture, flush render commands, and
    // signal CUDA via an imported timeline semaphore before returning.
    TSharedPtr<FJsonObject> R = MakeShared<FJsonObject>();
    R->SetBoolField(TEXT("ok"), true);
    return R;
}

TSharedPtr<FJsonObject> FJsonSocketServer::HandleShutdown(
    const TSharedPtr<FJsonObject>& Params, FString& OutError)
{
    TSharedPtr<FJsonObject> R = MakeShared<FJsonObject>();
    R->SetBoolField(TEXT("ok"), true);
    // Request engine exit after the response has been written.
    AsyncTask(ENamedThreads::GameThread, []() {
        FGenericPlatformMisc::RequestExit(false);
    });
    return R;
}
