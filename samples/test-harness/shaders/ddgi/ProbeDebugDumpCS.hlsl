/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef THGP_DIM_X
    #error Required define THGP_DIM_X is not defined for ProbeDebugDumpCS.hlsl!
#endif

#ifndef DDGI_PROBE_DEBUG_UAV_INDEX
    #error Required define DDGI_PROBE_DEBUG_UAV_INDEX is not defined for ProbeDebugDumpCS.hlsl!
#endif

#ifndef DDGI_PROBE_DEBUG_PASS_COUNT
    #error Required define DDGI_PROBE_DEBUG_PASS_COUNT is not defined for ProbeDebugDumpCS.hlsl!
#endif

#ifndef DDGI_PROBE_DEBUG_PASS_INDEX
    #error Required define DDGI_PROBE_DEBUG_PASS_INDEX is not defined for ProbeDebugDumpCS.hlsl!
#endif

#ifndef DDGI_PROBE_DEBUG_HEADER_ENTRIES
    #error Required define DDGI_PROBE_DEBUG_HEADER_ENTRIES is not defined for ProbeDebugDumpCS.hlsl!
#endif

#include "../include/Descriptors.hlsl"

#include "../../../../rtxgi-sdk/shaders/ddgi/include/DDGIRootConstants.hlsl"
#include "../../../../rtxgi-sdk/shaders/ddgi/include/ProbeCommon.hlsl"

#if RTXGI_BINDLESS_TYPE != RTXGI_BINDLESS_TYPE_DESCRIPTOR_HEAP
    #error ProbeDebugDumpCS currently supports descriptor heap bindless mode only.
#endif

[numthreads(THGP_DIM_X, 1, 1)]
void CS(uint3 DispatchThreadID : SV_DispatchThreadID)
{
    const uint ProbeDebugRecordsPerProbe = 3;

    uint volumeIndex = GetDDGIVolumeIndex();

    StructuredBuffer<DDGIVolumeDescGPUPacked> DDGIVolumes = GetDDGIVolumeConstants(GetDDGIVolumeConstantsIndex());
    StructuredBuffer<DDGIVolumeResourceIndices> DDGIVolumeBindless = GetDDGIVolumeResourceIndices(GetDDGIVolumeResourceIndicesIndex());
    RWStructuredBuffer<float4> ProbeDebugOut = ResourceDescriptorHeap[DDGI_PROBE_DEBUG_UAV_INDEX];

    DDGIVolumeResourceIndices resourceIndices = DDGIVolumeBindless[volumeIndex];
    DDGIVolumeDescGPU volume = UnpackDDGIVolumeDescGPU(DDGIVolumes[volumeIndex]);

    RWTexture2DArray<float4> ProbeIrradiance = GetRWTex2DArray(resourceIndices.probeIrradianceUAVIndex);
    RWTexture2DArray<float4> ProbeDistance = GetRWTex2DArray(resourceIndices.probeDistanceUAVIndex);
    RWTexture2DArray<float4> ProbeData = GetRWTex2DArray(resourceIndices.probeDataUAVIndex);

    uint totalProbes = volume.probeCounts.x * volume.probeCounts.y * volume.probeCounts.z;

    uint debugBufferEntries = 0;
    uint debugBufferStride = 0;
    ProbeDebugOut.GetDimensions(debugBufferEntries, debugBufferStride);

    uint maxProbeRecordsPerPass = 0;
    if (debugBufferEntries > DDGI_PROBE_DEBUG_HEADER_ENTRIES)
    {
        maxProbeRecordsPerPass = (debugBufferEntries - DDGI_PROBE_DEBUG_HEADER_ENTRIES) / (ProbeDebugRecordsPerProbe * DDGI_PROBE_DEBUG_PASS_COUNT);
    }

    uint probesToDump = min(totalProbes, maxProbeRecordsPerPass);

    if (DispatchThreadID.x == 0)
    {
        if (DDGI_PROBE_DEBUG_PASS_INDEX == 0)
        {
            ProbeDebugOut[0] = float4((float)totalProbes, (float)probesToDump, (float)volume.probeNumRays, (float)volumeIndex);
        }

        ProbeDebugOut[1] = float4((float)DDGI_PROBE_DEBUG_PASS_COUNT, (float)maxProbeRecordsPerPass, (float)ProbeDebugRecordsPerProbe, (float)DDGI_PROBE_DEBUG_PASS_INDEX);
    }

    uint probeIndex = DispatchThreadID.x;
    if (probeIndex >= probesToDump) return;

    int3 probeCoords = DDGIGetProbeCoords(probeIndex, volume);
    uint3 probeTexel = DDGIGetProbeTexelCoords(probeIndex, volume);

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

    uint segmentStride = maxProbeRecordsPerPass * ProbeDebugRecordsPerProbe;
    uint baseIndex = DDGI_PROBE_DEBUG_HEADER_ENTRIES + (DDGI_PROBE_DEBUG_PASS_INDEX * segmentStride) + (probeIndex * ProbeDebugRecordsPerProbe);
    ProbeDebugOut[baseIndex + 0] = float4((float3)probeCoords, probeState);
    ProbeDebugOut[baseIndex + 1] = float4(irradiance.rgb, distanceMoments.x);
    ProbeDebugOut[baseIndex + 2] = float4(probeOffset, distanceMoments.y);
}
