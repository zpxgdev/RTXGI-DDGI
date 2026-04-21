/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef DDGI_PROBE_DEBUG_RECORDS_ENABLED
    #define DDGI_PROBE_DEBUG_RECORDS_ENABLED 0
#endif

#ifndef DDGI_PROBE_DEBUG_PASS_COUNT
    #define DDGI_PROBE_DEBUG_PASS_COUNT 3
#endif

#ifndef DDGI_PROBE_DEBUG_RECORDS_PER_PROBE
    #define DDGI_PROBE_DEBUG_RECORDS_PER_PROBE 1
#endif

#ifndef DDGI_PROBE_DEBUG_HEADER_ENTRIES
    #define DDGI_PROBE_DEBUG_HEADER_ENTRIES 0
#endif

#ifndef DDGI_PROBE_DEBUG_MAX_VOLUMES
    #define DDGI_PROBE_DEBUG_MAX_VOLUMES 1
#endif

#ifndef DDGI_PROBE_DEBUG_PASS_INDEX
    #define DDGI_PROBE_DEBUG_PASS_INDEX 0
#endif

struct DDGIProbeDebugRecord
{
    uint4 header;
    uint4 layout;
    float4 probeMeta;
    float4 irradianceDistance;
    float4 probeOffsetDistance;
};

#if DDGI_PROBE_DEBUG_RECORDS_ENABLED && RTXGI_DDGI_BINDLESS_RESOURCES && (RTXGI_BINDLESS_TYPE == RTXGI_BINDLESS_TYPE_DESCRIPTOR_HEAP)

#ifndef DDGI_PROBE_DEBUG_UAV_INDEX
    #error Required define DDGI_PROBE_DEBUG_UAV_INDEX is not defined for DDGI probe debug record writes.
#endif

void DDGIWriteProbeDebugRecord(
    uint probeIndex,
    uint totalProbes,
    uint volumeIndex,
    DDGIVolumeDescGPU volume,
    RWTexture2DArray<float4> ProbeIrradiance,
    RWTexture2DArray<float4> ProbeDistance,
    RWTexture2DArray<float4> ProbeData)
{
    RWStructuredBuffer<DDGIProbeDebugRecord> ProbeDebugOut = ResourceDescriptorHeap[DDGI_PROBE_DEBUG_UAV_INDEX];

    uint debugBufferEntries = 0;
    uint debugBufferStride = 0;
    ProbeDebugOut.GetDimensions(debugBufferEntries, debugBufferStride);

    if (DDGI_PROBE_DEBUG_MAX_VOLUMES == 0) return;

    uint perVolumeEntries = debugBufferEntries / DDGI_PROBE_DEBUG_MAX_VOLUMES;
    if (perVolumeEntries == 0) return;

    uint maxProbeRecordsPerPass = perVolumeEntries / DDGI_PROBE_DEBUG_PASS_COUNT;
    uint probesToDump = min(totalProbes, maxProbeRecordsPerPass);

    uint volumeSlot = min(volumeIndex, DDGI_PROBE_DEBUG_MAX_VOLUMES - 1);
    uint volumeBase = volumeSlot * perVolumeEntries;

    if (probeIndex >= probesToDump) return;

    int3 probeCoords = DDGIGetProbeCoords((int)probeIndex, volume);
    uint3 probeTexel = DDGIGetProbeTexelCoords((int)probeIndex, volume);

    uint probeNumIrradianceTexels = uint(volume.probeNumIrradianceInteriorTexels + 2);
    uint probeNumDistanceTexels = uint(volume.probeNumDistanceInteriorTexels + 2);

    uint2 irradianceCenter = uint2((probeNumIrradianceTexels / 2), (probeNumIrradianceTexels / 2));
    uint2 distanceCenter = uint2((probeNumDistanceTexels / 2), (probeNumDistanceTexels / 2));

    uint2 irradianceTexel = uint2(probeTexel.x * probeNumIrradianceTexels, probeTexel.y * probeNumIrradianceTexels) + irradianceCenter;
    uint2 distanceTexel = uint2(probeTexel.x * probeNumDistanceTexels, probeTexel.y * probeNumDistanceTexels) + distanceCenter;

    float4 irradiance = ProbeIrradiance[uint3(irradianceTexel, probeTexel.z)];
    float2 distanceMoments = ProbeDistance[uint3(distanceTexel, probeTexel.z)].rg * 2.f;
    float4 probeData = ProbeData[probeTexel];

    float probeState = probeData.a;
    float3 probeOffset = DDGILoadProbeDataOffset(ProbeData, probeTexel, volume);

    uint segmentStride = maxProbeRecordsPerPass;
    uint baseIndex = volumeBase
        + (DDGI_PROBE_DEBUG_PASS_INDEX * segmentStride)
        + probeIndex;

    DDGIProbeDebugRecord record = (DDGIProbeDebugRecord)0;
    record.header = uint4(totalProbes, probesToDump, volume.probeNumRays, volumeIndex);
    record.layout = uint4(DDGI_PROBE_DEBUG_PASS_COUNT, maxProbeRecordsPerPass, DDGI_PROBE_DEBUG_RECORDS_PER_PROBE, DDGI_PROBE_DEBUG_PASS_INDEX);
    record.probeMeta = float4((float3)probeCoords, probeState);
    record.irradianceDistance = float4(irradiance.rgb, distanceMoments.x);
    record.probeOffsetDistance = float4(probeOffset, distanceMoments.y);

    ProbeDebugOut[baseIndex] = record;
}

#else

void DDGIWriteProbeDebugRecord(
    uint probeIndex,
    uint totalProbes,
    uint volumeIndex,
    DDGIVolumeDescGPU volume,
    RWTexture2DArray<float4> ProbeIrradiance,
    RWTexture2DArray<float4> ProbeDistance,
    RWTexture2DArray<float4> ProbeData)
{
}

#endif
