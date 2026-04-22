/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "Configs.h"
#include "Scenes.h"

#include <rtxgi/ddgi/DDGIVolume.h>
#include <fstream>

namespace SceneSyncExport
{
    bool ExportIfRequested(
        const Configs::Config& config,
        const Scenes::Scene& scene,
        const std::vector<rtxgi::DDGIVolumeDesc>& volumeDescs,
        const std::vector<rtxgi::DDGIVolumeBase*>& volumes,
        std::ofstream& log);
}
